"""CPU tests for building TensorSpec / CallSpec from torch tensors, alignment and aliasing (design 6.2, 6.5)."""

from __future__ import annotations

import pytest
import torch

from sol_kernels._runtime import bridge
from sol_kernels._runtime.interface import MAX_ALIGN, CallSpec, NotSupported, SpecError, TensorSpec


@pytest.mark.parametrize(
    ("ptr", "align"),
    [(0, 256), (1, 1), (2, 2), (6, 2), (12, 4), (4096, 256), (2**40, 256), (2**40 + 64, 64), (2**40 + 3, 1)],
)
def test_align_of_ptr(ptr, align):
    assert bridge.align_of_ptr(ptr) == align


def test_align_of_negative_pointer_rejected():
    with pytest.raises(ValueError):
        bridge.align_of_ptr(-8)


def base(n=4096, dtype=torch.float32):
    """A tensor whose data pointer is at least 256-byte aligned, so offsets below set the alignment."""
    t = bridge.alloc_with_layout(TensorSpec((n,), (1,), bridge.DTYPE_NAMES[dtype], 0, MAX_ALIGN), "cpu")
    assert t.data_ptr() % 256 == 0
    return t


def test_contiguous_spec():
    t = base()[:64].view(8, 8)
    s = bridge.tensor_spec(t)
    assert s.shape == (8, 8) and s.stride == (8, 1) and s.dtype == "f32"
    assert s.align_bytes == 256 and s.is_contiguous() and s.vec_bytes() == 16


@pytest.mark.parametrize(("off", "align"), [(1, 4), (2, 8), (3, 4), (4, 16), (8, 32), (64, 256)])
def test_storage_offset_sets_true_alignment(off, align):
    b = base()
    t = b[off:off + 16]
    s = bridge.tensor_spec(t)
    assert s.storage_offset == b.storage_offset() + off
    assert s.align_bytes == align
    assert s.vec_bytes() == min(16, align)


def test_alignment_of_bf16_offset_one():
    t = base(dtype=torch.bfloat16)[1:33]
    s = bridge.tensor_spec(t)
    assert s.align_bytes == 2 and s.vec_bytes() == 2


def test_layout_families():
    b = base()
    m = b[: 6 * 8].view(6, 8)
    assert bridge.tensor_spec(m.t()).stride == (1, 8)                      # permuted
    assert bridge.tensor_spec(m[:, ::2]).stride == (8, 2)                  # step-2 slice
    assert bridge.tensor_spec(b[:8].expand(5, 8)).stride == (0, 1)         # broadcast, stride 0 kept
    ov = b.as_strided((4, 8), (2, 1))                                      # overlapping rows
    assert bridge.tensor_spec(ov).stride == (2, 1) and not bridge.tensor_spec(ov).is_contiguous()
    cl = b[: 2 * 3 * 4 * 5].view(2, 3, 4, 5).contiguous(memory_format=torch.channels_last)
    assert bridge.tensor_spec(cl).stride == cl.stride()


def test_dtype_names_cover_target_dtypes():
    for dt, name in [(torch.bfloat16, "bf16"), (torch.float16, "f16"), (torch.float32, "f32"),
                     (torch.float64, "f64"), (torch.int8, "i8"), (torch.uint8, "u8"), (torch.bool, "bool"),
                     (torch.int64, "i64")]:
        assert bridge.tensor_spec(torch.zeros(4, dtype=dt)).dtype == name


@pytest.mark.parametrize(
    "make",
    [
        lambda: torch.zeros(4, dtype=torch.complex64),
        lambda: torch.zeros(4, dtype=torch.float8_e4m3fn),
        lambda: torch._neg_view(torch.zeros(4)),
        lambda: torch.zeros(4, 4).to_sparse(),
    ],
    ids=["complex", "fp8", "neg_view", "sparse"],
)
def test_unsupported_tensors(make):
    with pytest.raises(NotSupported):
        bridge.tensor_spec(make())


def test_tensor_subclass_rejected():
    class Sub(torch.Tensor):
        pass

    with pytest.raises(NotSupported):
        bridge.tensor_spec(torch.zeros(4).as_subclass(Sub))


# --- aliasing -----------------------------------------------------------------


def test_aliases():
    b = base()
    x, y = b[:16], b[1024:1040]
    assert bridge.aliases_of([x, y], [x]) == ((0, 0),)                   # in place
    assert bridge.aliases_of([x], [y]) == ()                               # disjoint
    assert bridge.aliases_of([b[:16]], [b[8:24]]) == ((0, 0),)             # partial overlap
    assert bridge.aliases_of([b[:16]], [b[16:32]]) == ()                   # adjacent, no overlap
    assert bridge.aliases_of([b[0:0]], [b[0:16]]) == ()                    # empty tensors alias nothing
    m = b[:64].view(8, 8)
    assert bridge.aliases_of([m[:, 0::2]], [m[:, 1::2]]) == ((0, 0),)      # interleaved: reported (conservative)


def test_call_spec_building():
    b = base()
    x, y = b[:32].view(4, 8), b[1024:1056].view(4, 8)
    out = b[2048:2080].view(4, 8)
    s = bridge.call_spec("add", [x, y], [out], attrs=[("dim", 1)], scalars=[1.5, 2, True])
    assert isinstance(s, CallSpec)
    assert s.device == "cpu" and s.scalar_kinds == ("float", "int", "bool") and s.aliases == ()
    assert s == bridge.call_spec("add", [x, y], [out], attrs=[("dim", 1)], scalars=[0.0, -7, False])
    assert s != bridge.call_spec("add", [x, y], [x], attrs=[("dim", 1)], scalars=[1.5, 2, True])  # in place
    assert s != bridge.call_spec("add", [x, b[1:33].view(4, 8)], [out], attrs=[("dim", 1)], scalars=[1.5, 2, True])
    assert bridge.call_spec("add", [x], [out], device="X|sm120|1|2").device == "X|sm120|1|2"


@pytest.mark.parametrize("bad", [torch.tensor(1.0), None, "1", 1 + 2j])
def test_bad_scalars_rejected(bad):
    with pytest.raises(SpecError):
        bridge.call_spec("add", [torch.zeros(4)], [torch.zeros(4)], scalars=[bad])


def test_mixed_devices_rejected():
    meta = torch.zeros(4, device="meta")
    with pytest.raises(SpecError):
        bridge.call_spec("add", [torch.zeros(4)], [meta])


# --- dummies --------------------------------------------------------------------


SPECS = [
    TensorSpec((4, 8), (8, 1), "f32", 0, 256),
    TensorSpec((4, 8), (8, 1), "f32", 1, 4),
    TensorSpec((4, 8), (8, 1), "bf16", 3, 2),
    TensorSpec((4, 8), (1, 4), "f16", 0, 64),
    TensorSpec((5, 8), (0, 1), "f32", 0, 256),
    TensorSpec((4, 8), (2, 1), "f64", 0, 8),
    TensorSpec((0, 8), (8, 1), "f32", 0, 256),
    TensorSpec((), (), "i32", 0, 16),
    TensorSpec((7,), (1,), "u8", 5, 1),
    TensorSpec((7,), (3,), "bool", 0, 32),
]


@pytest.mark.parametrize("spec", SPECS, ids=str)
def test_alloc_with_layout(spec):
    t = bridge.alloc_with_layout(spec, "cpu")
    got = bridge.tensor_spec(t)
    assert (got.shape, got.stride, got.dtype) == (spec.shape, spec.stride, spec.dtype)
    assert t.data_ptr() % spec.align_bytes == 0
    assert bridge.is_compatible(t, spec)
    if spec.numel and t.dtype.is_floating_point:
        assert torch.isnan(t).all()


def test_dummies_share_memory_only_for_in_place():
    x = TensorSpec((4, 8), (8, 1), "f32", 0, 256)
    s = CallSpec("add", (x, x), (x,), (), (), ((0, 1),), "cpu")
    ins, outs, ws = bridge.make_dummies(s, "cpu", workspace_bytes=100)
    assert outs[0].data_ptr() == ins[1].data_ptr() != ins[0].data_ptr()
    assert ws.dtype == torch.uint8 and ws.numel() == 100 and ws.data_ptr() % 256 == 0
    s2 = CallSpec("add", (x, x), (x,), (), (), (), "cpu")
    ins, outs, ws = bridge.make_dummies(s2, "cpu")
    assert outs[0].data_ptr() not in (ins[0].data_ptr(), ins[1].data_ptr()) and ws is None


def test_is_compatible_checks_alignment_promise():
    spec = TensorSpec((16,), (1,), "f32", 0, 16)
    b = base()
    assert bridge.is_compatible(b[:16], spec)
    assert not bridge.is_compatible(b[1:17], spec)          # only 4-byte aligned
    assert not bridge.is_compatible(b[:32:2], spec)         # strides differ


def test_to_cute_on_cpu_uses_the_spec_alignment():
    spec = TensorSpec((4, 8), (8, 1), "f32", 0, 16)
    t = bridge.alloc_with_layout(spec, "cpu")
    c = bridge.to_cute(t, spec)
    assert tuple(c.shape) == (4, 8)
    with pytest.raises(SpecError):
        bridge.to_cute(t.t(), spec)
