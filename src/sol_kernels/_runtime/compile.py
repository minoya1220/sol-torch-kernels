"""sol_compile: the only way impls compile kernels (design 2.3).

Wraps `cutlass.cute.compile` and records, per compiled module, the names of its
device functions and hashes of its CUBIN and PTX. The kernel allowlist for a run
is exactly the set of names recorded here; the module hashes let the trace check
(step 7) tie each launched kernel to a module this runtime produced.

Kernel names come from the DSL's own API: `JitCompiledFunction.kernel_info`, whose
keys are the device symbols (nvidia-cutlass-dsl 4.8.1). Each name is also checked
against the CUBIN. With that DSL version the symbol of a `@cute.kernel` function
`sol_copy_kernel` is `kernel_cutlass_sol_copy_kernel_<argument mangling>_<n>`, so a
device name starts with `kernel_cutlass_` and the lint's `sol_` rule on the Python
name shows up as the prefix `kernel_cutlass_sol_`. Recording fails closed: if the
DSL gives no names, no CUBIN, a name outside that prefix, or a name missing from
the CUBIN, compilation raises instead of producing an unrecorded module.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

KERNEL_NAME_PREFIX = "kernel_cutlass_sol_"


class KernelRecordError(RuntimeError):
    """The compiled module's kernels could not be recorded, so it is not allowed."""


@dataclass(frozen=True)
class CompiledModule:
    impl: str                       # who asked for it: "module:qualname" of the Impl
    kernel_names: tuple[str, ...]   # device symbols in the module
    cubin_sha256: str
    ptx_sha256: str | None
    arch: str | None                # --gpu-arch passed to the DSL, if any


class CompileRecorder:
    """Process-wide record of everything sol_compile produced."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._modules: list[CompiledModule] = []

    def add(self, m: CompiledModule) -> None:
        with self._lock:
            self._modules.append(m)

    def modules(self) -> tuple[CompiledModule, ...]:
        with self._lock:
            return tuple(self._modules)

    def allowlist(self) -> frozenset[str]:
        return frozenset(n for m in self.modules() for n in m.kernel_names)

    def module_hashes(self) -> frozenset[str]:
        return frozenset(m.cubin_sha256 for m in self.modules())


RECORDER = CompileRecorder()

_DEVICE_ARCH = re.compile(r"\|sm(\d+)\|")


def arch_from_device(device: str) -> str | None:
    """'NVIDIA GeForce RTX 5090|sm120|170|100663296' -> 'sm_120a'. None for a key
    with no compute capability (for example 'cpu'), which leaves the DSL's default."""
    m = _DEVICE_ARCH.search(device)
    return f"sm_{m.group(1)}a" if m else None


def sol_compile(
    fn: Callable[..., Any],
    *args: Any,
    impl: str,
    arch: str | None = None,
    recorder: CompileRecorder | None = None,
    **kwargs: Any,
) -> Any:
    """Compile `fn` (a `@cute.jit` function) for `args` and record its kernels.

    Impls may not pass compile options: an option string can link external
    libraries or change numerics, which an impl does not get to choose.
    """
    if "options" in kwargs:
        raise ValueError("impls may not pass compile options to sol_compile")
    if arch is not None and not re.fullmatch(r"sm_\d+a?", arch):
        raise ValueError(f"bad arch {arch!r}")
    import cutlass.cute as cute  # deferred: keeps `import sol_kernels._runtime` light

    # The DSL writes kept CUBIN and PTX files to a dump directory (the current
    # directory by default). Use a private temporary one and delete it.
    dump_dir = tempfile.mkdtemp(prefix="sol_compile_")
    try:
        options = f"--keep-cubin --keep-ptx --dump-dir {dump_dir}" + (f" --gpu-arch {arch}" if arch else "")
        compiled = cute.compile(fn, *args, options=options, **kwargs)
        names = tuple(getattr(compiled, "kernel_info", None) or ())
        cubin = getattr(compiled, "__cubin__", None)
        ptx = getattr(compiled, "__ptx__", None)
    finally:
        shutil.rmtree(dump_dir, ignore_errors=True)
    if not names:
        raise KernelRecordError(f"{impl}: the DSL reported no kernels for {getattr(fn, '__name__', fn)!r}")
    if not isinstance(cubin, bytes) or not cubin:
        raise KernelRecordError(f"{impl}: the DSL kept no CUBIN, so the module cannot be hashed")
    bad = [n for n in names if not n.startswith(KERNEL_NAME_PREFIX)]
    if bad:
        raise KernelRecordError(f"{impl}: kernel names without the {KERNEL_NAME_PREFIX!r} prefix: {bad}")
    missing = [n for n in names if n.encode() not in cubin]
    if missing:
        raise KernelRecordError(f"{impl}: kernel names not found in the CUBIN: {missing}")

    (recorder if recorder is not None else RECORDER).add(CompiledModule(
        impl=impl,
        kernel_names=names,
        cubin_sha256=hashlib.sha256(cubin).hexdigest(),
        ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest() if isinstance(ptx, str) else None,
        arch=arch,
    ))
    return compiled
