"""GPU test of the honest copy impl through the runtime (implementation step 3, "GPU").

Needs an sm120 GPU. Skipped everywhere else, including CPU CI; it must pass in the
external sm120 run before this path is trusted. Not run by the author of this file.
"""

from __future__ import annotations

import pytest
import torch

from sol_kernels._runtime import bridge, compile as rc, dispatch
from sol_kernels._runtime.interface import Registry, TensorSpec
from _load import load_copy_honest


def _is_sm120() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)


pytestmark = [
    pytest.mark.sm120,
    pytest.mark.skipif(not _is_sm120(), reason="needs an sm120 GPU: runs in the external sm120 run"),
]

DTYPES = [torch.float32, torch.bfloat16, torch.float16, torch.float64, torch.int8, torch.uint8, torch.int64, torch.bool]
SHAPES = [(1,), (7,), (3, 5, 7), (63, 4099), (1, 2**20), (2**20, 1), (255,), (257,)]


def _random_like_spec(ts: TensorSpec) -> torch.Tensor:
    t = bridge.alloc_with_layout(ts, "cuda", poison=False)
    raw = t.view(torch.uint8) if t.is_contiguous() else None
    if raw is not None and t.numel():
        raw.copy_(torch.randint(0, 256, raw.shape, dtype=torch.uint8, device="cuda"))  # every bit pattern, NaNs included
    return t


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
@pytest.mark.parametrize("off", [0, 1, 3])
def test_copy_is_bitwise_exact(dtype, shape, off):
    name = bridge.DTYPE_NAMES[dtype]
    isz = torch.empty(0, dtype=dtype).element_size()
    stride, acc = [], 1
    for s in reversed(shape):
        stride.insert(0, acc)
        acc *= s
    align = bridge.align_of_ptr(off * isz) if off else 256
    src_spec = TensorSpec(shape, tuple(stride), name, off, align)
    dst_spec = TensorSpec(shape, tuple(stride), name, 0, 256)
    impl = load_copy_honest().CopyHonest()
    rec = rc.CompileRecorder()
    first = True
    launcher = None
    for _ in range(3):  # calls 2..n must also do the work (design D4)
        x = _random_like_spec(src_spec)
        y = bridge.alloc_with_layout(dst_spec, "cuda")  # poisoned
        spec = bridge.call_spec("copy", [x], [y])
        if first:
            launcher = dispatch.make_launcher(impl, spec, recorder=rec)
            first = False
        launcher([x], [y], None, ())
        torch.cuda.synchronize()
        assert torch.equal(_bits(y), _bits(x))
    names = rec.allowlist()
    assert len(names) == 1 and next(iter(names)).startswith(rc.KERNEL_NAME_PREFIX)


def test_public_path_on_current_stream():
    reg = Registry()
    reg.register(load_copy_honest().CopyHonest)
    cache = dispatch.LauncherCache()
    s = torch.cuda.Stream()
    x = torch.randn(4097, device="cuda")
    y = torch.full_like(x, float("nan"))
    with torch.cuda.stream(s):
        dispatch.call("copy", [x], [y], registry=reg, cache=cache)
    s.synchronize()
    assert torch.equal(x, y)


def test_launcher_refuses_a_different_alignment():
    impl = load_copy_honest().CopyHonest()
    b = torch.empty(4096, device="cuda")
    x, y = b[:1024], torch.empty(1024, device="cuda")
    L = dispatch.make_launcher(impl, bridge.call_spec("copy", [x], [y]))
    with pytest.raises(dispatch.SpecMismatch):
        L([b[1:1025]], [y], None, ())
