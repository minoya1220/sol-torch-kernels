"""torch tensor <-> TensorSpec / cute tensor, alignment, aliasing, streams (design 6.2, 6.5).

Protected path. Everything here except `device_key`, `to_cute` on device tensors and
`current_stream` works on CPU tensors, which is how the unit tests exercise it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .interface import MAX_ALIGN, CallSpec, NotSupported, SpecError, TensorSpec

DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bool: "bool",
    torch.uint8: "u8", torch.int8: "i8",
    torch.uint16: "u16", torch.int16: "i16",
    torch.uint32: "u32", torch.int32: "i32",
    torch.uint64: "u64", torch.int64: "i64",
    torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32", torch.float64: "f64",
}
TORCH_DTYPES: dict[str, torch.dtype] = {v: k for k, v in DTYPE_NAMES.items()}


def align_of_ptr(ptr: int) -> int:
    """Largest power of two dividing ptr, capped at MAX_ALIGN. A null pointer (an
    empty tensor) counts as fully aligned."""
    if ptr < 0:
        raise ValueError(f"negative pointer {ptr}")
    if ptr == 0:
        return MAX_ALIGN
    return min(ptr & -ptr, MAX_ALIGN)


def _check_plain(t: Any) -> None:
    if type(t) not in (torch.Tensor, torch.nn.Parameter):
        raise NotSupported(f"only plain torch.Tensor is supported, got {type(t).__name__}")
    if t.layout != torch.strided:
        raise NotSupported(f"only strided tensors are supported, got {t.layout}")
    if t.is_quantized:
        raise NotSupported("quantized tensors are not supported")
    # Lazy negation / conjugation views read as transformed values while the memory
    # holds the originals; a kernel reading the memory would get the wrong values.
    if t.is_neg() or t.is_conj():
        raise NotSupported("tensors with a pending neg or conj bit are not supported; call .resolve_neg()/.resolve_conj()")
    if t.dtype not in DTYPE_NAMES:
        raise NotSupported(f"dtype {t.dtype} is not supported")


def tensor_spec(t: torch.Tensor) -> TensorSpec:
    _check_plain(t)
    return TensorSpec(
        shape=tuple(int(s) for s in t.shape),
        stride=tuple(int(s) for s in t.stride()),
        dtype=DTYPE_NAMES[t.dtype],
        storage_offset=int(t.storage_offset()),
        align_bytes=align_of_ptr(t.data_ptr()),
    )


def byte_extent(t: torch.Tensor) -> tuple[int, int] | None:
    """[lo, hi) byte range of memory the view can touch; None when it has no elements."""
    if t.numel() == 0:
        return None
    span = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    lo = t.data_ptr()
    return lo, lo + (span + 1) * t.element_size()


def aliases_of(inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor]) -> tuple[tuple[int, int], ...]:
    """(output index, input index) for every pair whose memory ranges overlap. This
    is conservative: two strided views can interleave without sharing an element,
    and they are still reported, so the impl must treat them as aliasing."""
    pairs = []
    in_ext = [byte_extent(t) for t in inputs]
    for o, out in enumerate(outputs):
        oe = byte_extent(out)
        if oe is None:
            continue
        for i, ie in enumerate(in_ext):
            if ie is not None and oe[0] < ie[1] and ie[0] < oe[1]:
                pairs.append((o, i))
    return tuple(pairs)


def scalar_kind(v: Any) -> str:
    if type(v) is bool:
        return "bool"
    if type(v) is int:
        return "int"
    if type(v) is float:
        return "float"
    raise SpecError(f"run-time scalars must be Python bool, int or float, got {type(v).__name__}")


def device_key(device: torch.device) -> str:
    """Tuning device key (design 5.2): '{name}|sm{cc}|{sm_count}|{l2_bytes}'.

    Uses the CUDA device name; design 5.2 says the NVML name. For GeForce cards the
    two are the same string; to confirm on the sm120 run.
    """
    if device.type != "cuda":
        return device.type
    p = torch.cuda.get_device_properties(device)
    return f"{p.name}|sm{p.major}{p.minor}|{p.multi_processor_count}|{p.L2_cache_size}"


def call_spec(
    op: str,
    inputs: Sequence[torch.Tensor],
    outputs: Sequence[torch.Tensor],
    *,
    attrs: Sequence[tuple[str, Any]] = (),
    scalars: Sequence[Any] = (),
    device: str | None = None,
) -> CallSpec:
    """Build the CallSpec of a call from its tensors. `device` overrides the key
    derived from the tensors (tests use it to build sm120 specs on CPU)."""
    tensors = [*inputs, *outputs]
    devices = {t.device for t in tensors}
    if len(devices) > 1:
        raise SpecError(f"tensors on more than one device: {sorted(map(str, devices))}")
    if device is None:
        device = device_key(devices.pop()) if devices else "none"
    return CallSpec(
        op=op,
        inputs=tuple(tensor_spec(t) for t in inputs),
        outputs=tuple(tensor_spec(t) for t in outputs),
        attrs=tuple((str(k), v) for k, v in attrs),
        scalar_kinds=tuple(scalar_kind(s) for s in scalars),
        aliases=aliases_of(inputs, outputs),
        device=device,
    )


# --- dummies for compilation ---------------------------------------------------


def _poison_(base: torch.Tensor) -> None:
    """NaN for float storage, random bytes for everything else (design 4.2 step 2)."""
    if base.dtype.is_floating_point:
        base.fill_(float("nan"))
    else:
        raw = base.view(torch.uint8)
        raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8, device=raw.device))


def alloc_with_layout(ts: TensorSpec, device: torch.device | str, *, poison: bool = True) -> torch.Tensor:
    """Fresh storage viewed with exactly `ts`'s shape, strides and dtype, with the
    data pointer aligned to at least `ts.align_bytes`. Stride 0 is preserved (no
    clone(), which would densify a broadcast view). The storage offset of the result
    is chosen to meet the alignment, so it can differ from `ts.storage_offset`."""
    dtype = TORCH_DTYPES[ts.dtype]
    isz = ts.itemsize
    span = sum((s - 1) * st for s, st in zip(ts.shape, ts.stride)) + 1 if ts.numel else 0
    pad = ts.align_bytes // isz
    base = torch.empty(ts.storage_offset + span + pad, dtype=dtype, device=device)
    if poison and base.numel():
        _poison_(base)
    start = base.data_ptr() + ts.storage_offset * isz
    shift_bytes = (-start) % ts.align_bytes
    if shift_bytes % isz:
        raise SpecError(f"cannot align a {ts.dtype} view to {ts.align_bytes} bytes")
    return base.as_strided(ts.shape, ts.stride, ts.storage_offset + shift_bytes // isz)


def is_compatible(t: torch.Tensor, ts: TensorSpec) -> bool:
    """Same shape, strides and dtype, and the pointer keeps ts's alignment promise."""
    got = tensor_spec(t)
    return (
        got.shape == ts.shape and got.stride == ts.stride and got.dtype == ts.dtype
        and got.align_bytes >= ts.align_bytes
    )


def make_dummies(
    spec: CallSpec, device: torch.device | str, workspace_bytes: int = 0
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Poisoned tensors with the spec's layouts, for compilation only. An output
    that is declared to alias an input with an identical spec (in-place) shares
    that input's dummy; other outputs get their own memory."""
    ins = tuple(alloc_with_layout(ts, device) for ts in spec.inputs)
    outs = []
    for o, ts in enumerate(spec.outputs):
        same = [i for (oo, i) in spec.aliases if oo == o and spec.inputs[i] == ts]
        outs.append(ins[same[0]] if same else alloc_with_layout(ts, device))
    ws = None
    if workspace_bytes:
        ws = alloc_with_layout(TensorSpec((workspace_bytes,), (1,), "u8", 0, MAX_ALIGN), device)
    return ins, tuple(outs), ws


# --- cute side -----------------------------------------------------------------


def to_cute(t: torch.Tensor, ts: TensorSpec) -> Any:
    """Convert with the spec's true alignment (design 6.5), never a fixed 16. The
    tensor must be compatible with the spec the launcher was built for."""
    if not is_compatible(t, ts):
        raise SpecError(f"tensor {tensor_spec(t)} does not match the built spec {ts}")
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t, assumed_align=ts.align_bytes)


def current_stream() -> Any:
    """The current torch CUDA stream as the DSL's stream type. Needs a GPU."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)


def placeholder_stream() -> Any:
    """A stream value to compile against (the null stream). No GPU needed."""
    import cuda.bindings.driver as cuda

    return cuda.CUstream(0)
