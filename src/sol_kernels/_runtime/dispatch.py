"""Building launchers and calling them (design 6.3, 6.5).

Two paths:

- Harness path: `make_launcher(impl, spec, cfg)` builds one launcher for one spec.
  It never falls back: an unsupported spec raises `NotSupported`. The harness owns
  all allocation (slots, red zones, poison) and calls the returned `BoundLauncher`.
- Public path: `call(op, inputs, outputs, ...)` builds the spec from the tensors,
  looks up the launcher cache and launches on the current stream. Outside
  evaluation it may fall back to a torch function the caller passes (open question
  10.8). Output allocation stays with the per-op public wrappers, which do not
  exist yet.

The runtime owns the spec: a `BoundLauncher` recomputes the CallSpec of every call
from the tensors it is given and refuses any call that differs from the spec it
was built for (shape, strides, dtype, storage offset, pointer alignment, aliasing,
scalar kinds, device). That is the fix for the old wrappers, which compiled a
static layout and then accepted any tensor.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from . import bridge
from .compile import RECORDER, CompileRecorder, arch_from_device, sol_compile
from .interface import REGISTRY, CallSpec, Config, Impl, NotSupported, Registry


class SpecMismatch(RuntimeError):
    """A launcher was called with tensors whose spec differs from the built one."""


class ImplContractError(RuntimeError):
    """An impl broke the interface contract (design 6.3)."""


def impl_name(impl: Impl) -> str:
    cls = type(impl)
    return f"{cls.__module__}:{cls.__qualname__}"


class RuntimeBuildContext:
    """The BuildContext an impl's `build` receives. Usable only during build."""

    def __init__(
        self,
        spec: CallSpec,
        impl: str,
        *,
        torch_device: torch.device | str,
        workspace_bytes: int = 0,
        recorder: CompileRecorder | None = None,
    ) -> None:
        self._spec = spec
        self._impl = impl
        self._torch_device = torch_device
        self._workspace_bytes = workspace_bytes
        self._recorder = recorder if recorder is not None else RECORDER
        self._open = True
        self._dummies: tuple | None = None

    def _check_open(self) -> None:
        if not self._open:
            raise ImplContractError("BuildContext used after build returned")

    def compile(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        self._check_open()
        return sol_compile(
            fn, *args, impl=self._impl, arch=arch_from_device(self._spec.device),
            recorder=self._recorder, **kwargs,
        )

    def dummies(self) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
        self._check_open()
        if self._dummies is None:
            ins, outs, ws = bridge.make_dummies(self._spec, self._torch_device, self._workspace_bytes)
            self._torch_dummies = (ins, outs, ws)
            self._dummies = (
                tuple(bridge.to_cute(t, ts) for t, ts in zip(ins, self._spec.inputs)),
                tuple(bridge.to_cute(t, ts) for t, ts in zip(outs, self._spec.outputs)),
                None if ws is None else bridge.to_cute(ws, bridge.tensor_spec(ws)),
            )
        return self._dummies

    def stream(self) -> Any:
        self._check_open()
        return bridge.placeholder_stream()

    def close(self) -> None:
        """Drop every reference to the dummies (design 6.3: freed after build)."""
        self._open = False
        self._dummies = None
        self._torch_dummies = None


def _check_config(impl: Impl, spec: CallSpec, cfg: Config) -> dict[str, int]:
    space = type(impl).search_space
    if set(cfg) != set(space):
        raise ValueError(f"config keys {sorted(cfg)} differ from the search space {sorted(space)}")
    for k, v in cfg.items():
        if type(v) is not int or v not in space[k].options:
            raise ValueError(f"config {k}={v!r} is not one of {space[k].options}")
    if not impl.valid_config(spec, dict(cfg)):
        raise ValueError(f"config {cfg} is not valid for this spec")
    return dict(cfg)


@dataclass
class BoundLauncher:
    impl: str
    spec: CallSpec
    cfg: dict[str, int]
    workspace_bytes: int
    _fn: Callable[..., Any] = field(repr=False)

    def __call__(
        self,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        workspace: torch.Tensor | None = None,
        scalars: Sequence[Any] = (),
        stream: Any = None,
    ) -> None:
        got = bridge.call_spec(self.spec.op, inputs, outputs, attrs=self.spec.attrs, scalars=scalars)
        if got != self.spec:
            raise SpecMismatch(f"{self.impl}: call spec {got} differs from the built spec {self.spec}")
        ws = None
        if self.workspace_bytes:
            if workspace is None or workspace.dtype != torch.uint8 or not workspace.is_contiguous() \
                    or workspace.numel() < self.workspace_bytes or workspace.device != inputs[0].device:
                raise SpecMismatch(f"{self.impl}: needs a contiguous uint8 workspace of {self.workspace_bytes} bytes")
            ws = bridge.to_cute(workspace, bridge.tensor_spec(workspace))
        elif workspace is not None:
            raise SpecMismatch(f"{self.impl}: declared no workspace but was given one")
        ins = tuple(bridge.to_cute(t, ts) for t, ts in zip(inputs, self.spec.inputs))
        outs = tuple(bridge.to_cute(t, ts) for t, ts in zip(outputs, self.spec.outputs))
        if stream is None:
            stream = bridge.current_stream()
        ret = self._fn(ins, outs, ws, tuple(scalars), stream)
        if ret is not None:
            raise ImplContractError(f"{self.impl}: launcher returned {type(ret).__name__}, must return None")


def make_launcher(
    impl: Impl,
    spec: CallSpec,
    cfg: Config | None = None,
    *,
    torch_device: torch.device | str | None = None,
    recorder: CompileRecorder | None = None,
) -> BoundLauncher:
    """Harness path. Never falls back."""
    if not impl.supports(spec):
        raise NotSupported(f"{impl_name(impl)} does not support {spec.digest()[:12]}")
    cfg = _check_config(impl, spec, impl.default_config(spec) if cfg is None else cfg)
    ws = impl.workspace_bytes(spec, dict(cfg))
    if type(ws) is not int or ws < 0:
        raise ImplContractError(f"{impl_name(impl)}: workspace_bytes returned {ws!r}")
    if torch_device is None:
        torch_device = "cuda" if spec.device not in ("cpu", "none") else "cpu"
    ctx = RuntimeBuildContext(spec, impl_name(impl), torch_device=torch_device,
                              workspace_bytes=ws, recorder=recorder)
    try:
        fn = impl.build(spec, dict(cfg), ctx)
    finally:
        ctx.close()
    if not callable(fn):
        raise ImplContractError(f"{impl_name(impl)}: build returned {type(fn).__name__}, not a launcher")
    return BoundLauncher(impl=impl_name(impl), spec=spec, cfg=cfg, workspace_bytes=ws, _fn=fn)


class LauncherCache:
    """(impl class, CallSpec, config) -> BoundLauncher, so each spec builds once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[tuple, BoundLauncher] = {}

    def get(self, impl: Impl, spec: CallSpec, cfg: Config, **kw: Any) -> BoundLauncher:
        key = (type(impl), spec, tuple(sorted(cfg.items())))
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        built = make_launcher(impl, spec, cfg, **kw)
        with self._lock:
            return self._cache.setdefault(key, built)

    def __len__(self) -> int:
        return len(self._cache)


CACHE = LauncherCache()


def call(
    op: str,
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    *,
    attrs: Sequence[tuple[str, Any]] = (),
    scalars: Sequence[Any] = (),
    fallback: Callable[[], Any] | None = None,
    registry: Registry | None = None,
    cache: LauncherCache | None = None,
    stream: Any = None,
) -> None:
    """Public path. Config: the impl's default (the tuning table lookup comes with
    the tuner, implementation step 10)."""
    cls = (registry if registry is not None else REGISTRY).get(op)
    spec = bridge.call_spec(op, inputs, outputs, attrs=attrs, scalars=scalars)
    impl = cls() if cls is not None else None
    if impl is None or not impl.supports(spec):
        if fallback is not None:
            fallback()
            return
        raise NotSupported(f"no impl of {op!r} supports this call")
    cfg = impl.default_config(spec)
    launcher = (cache if cache is not None else CACHE).get(impl, spec, cfg)
    ws = None
    if launcher.workspace_bytes:
        ws = torch.empty(launcher.workspace_bytes, dtype=torch.uint8, device=inputs[0].device)
    launcher(inputs, outputs, ws, scalars, stream)
