"""CPU tests for TensorSpec, CallSpec (building, hashing), Param and the registry (design 6.2, 6.3)."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from sol_kernels._runtime.interface import (
    INTERFACE_VERSION,
    CallSpec,
    Impl,
    Param,
    Registry,
    SpecError,
    TensorSpec,
    register_impl,
)

DEV = "NVIDIA GeForce RTX 5090|sm120|170|100663296"


def ts(shape=(4, 8), stride=None, dtype="f32", off=0, align=256):
    if stride is None:
        stride, acc = [], 1
        for s in reversed(shape):
            stride.insert(0, acc)
            acc *= s
        stride = tuple(stride)
    return TensorSpec(tuple(shape), tuple(stride), dtype, off, align)


def cs(**kw):
    base = dict(op="add", inputs=(ts(), ts()), outputs=(ts(),), attrs=(), scalar_kinds=(), aliases=(), device=DEV)
    base.update(kw)
    return CallSpec(**base)


def test_interface_version():
    assert INTERFACE_VERSION == 1


# --- TensorSpec validation -------------------------------------------------


@pytest.mark.parametrize(
    "kw",
    [
        dict(shape=[4], stride=(1,)),          # list, not tuple
        dict(shape=(4,), stride=(1, 1)),       # rank mismatch
        dict(shape=(-1,), stride=(1,)),
        dict(shape=(4,), stride=(-1,)),
        dict(shape=(True,), stride=(1,)),      # bool is not an int here
        dict(dtype="f8"),
        dict(dtype="complex64"),
        dict(off=-1),
        dict(align=3),
        dict(align=512),
        dict(align=0),
        dict(dtype="f32", align=2),            # pointer not aligned to its own element size
    ],
)
def test_tensorspec_rejects(kw):
    args = dict(shape=(4,), stride=(1,), dtype="f32", off=0, align=16)
    args.update(kw)
    with pytest.raises(SpecError):
        TensorSpec(args["shape"], args["stride"], args["dtype"], args["off"], args["align"])


@pytest.mark.parametrize(
    ("shape", "stride", "contig"),
    [
        ((4, 8), (8, 1), True),
        ((4, 8), (1, 4), False),
        ((4, 8), (16, 2), False),
        ((4, 1), (1, 7), True),       # size-1 dims may have any stride
        ((1, 8), (99, 1), True),
        ((0, 8), (5, 3), True),       # empty
        ((), (), True),
        ((4, 8), (0, 1), False),      # broadcast
        ((3, 4), (2, 1), False),      # overlapping rows
    ],
)
def test_is_contiguous(shape, stride, contig):
    assert ts(shape, stride).is_contiguous() is contig


@pytest.mark.parametrize(
    ("spec", "vec"),
    [
        (ts((4, 8), dtype="f32", align=256), 16),
        (ts((4, 8), dtype="bf16", align=256), 16),
        (ts((4, 8), dtype="f32", align=8), 8),        # alignment limits
        (ts((4, 8), dtype="f32", off=1, align=4), 4),
        (ts((4, 3), dtype="f32", align=256), 4),      # row of 12 bytes
        (ts((4, 6), dtype="f32", align=256), 8),      # row of 24 bytes
        (ts((4, 8), (16, 1), "f32", 0, 256), 16),     # sliced rows, 64-byte row stride
        (ts((4, 8), (10, 1), "f32", 0, 256), 8),      # 40-byte row stride
        (ts((4, 8), (1, 4), "f32", 0, 256), 4),       # transposed: inner dim not unit stride
        (ts((4, 8), (0, 1), "f32", 0, 256), 16),      # broadcast outer dim: rows identical, still aligned
        (ts((4, 8), (8, 0), "f32", 0, 256), 4),       # broadcast inner dim
        (ts((), (), "f64", 0, 256), 8),
        (ts((0, 8), dtype="f32"), 4),
        (ts((1, 1024), dtype="u8", align=256), 16),
        (ts((4, 1), (1, 1), "f16", 0, 256), 2),       # inner size 1
    ],
)
def test_vec_bytes(spec, vec):
    assert spec.vec_bytes() == vec


# --- CallSpec building and hashing ------------------------------------------


def test_callspec_is_hashable_and_a_dict_key():
    d = {cs(): 1}
    assert d[cs()] == 1
    assert cs() == cs() and hash(cs()) == hash(cs())


def test_attrs_are_canonicalized_by_name():
    a = cs(attrs=(("keepdim", False), ("dim", 1)))
    b = cs(attrs=(("dim", 1), ("keepdim", False)))
    assert a == b and hash(a) == hash(b) and a.digest() == b.digest()
    assert a.attrs == (("dim", 1), ("keepdim", False))
    assert a.attr("dim") == 1
    with pytest.raises(KeyError):
        a.attr("missing")


@pytest.mark.parametrize(("x", "y"), [(True, 1), (1, 1.0), (0.0, -0.0), (False, 0), ("1", 1)])
def test_attr_types_are_part_of_identity(x, y):
    a, b = cs(attrs=(("k", x),)), cs(attrs=(("k", y),))
    assert a != b and a.digest() != b.digest()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, (1, 2), b"x"])
def test_bad_attr_values_rejected(bad):
    with pytest.raises(SpecError):
        cs(attrs=(("k", bad),))


def test_duplicate_attr_names_rejected():
    with pytest.raises(SpecError):
        cs(attrs=(("dim", 0), ("dim", 1)))


@pytest.mark.parametrize(
    "kw",
    [
        dict(aliases=((1, 0),)),          # no output 1
        dict(aliases=((0, 2),)),          # no input 2
        dict(aliases=((0, 0), (0, 0))),
        dict(scalar_kinds=("complex",)),
        dict(device=""),
        dict(op=""),
        dict(inputs=[ts()]),
    ],
)
def test_callspec_rejects(kw):
    with pytest.raises(SpecError):
        cs(**kw)


def test_aliases_canonicalized():
    a = cs(inputs=(ts(), ts()), aliases=((0, 1), (0, 0)))
    assert a.aliases == ((0, 0), (0, 1))


VARIANTS = [
    dict(op="mul"),
    dict(inputs=(ts(), ts((4, 8), (1, 4)))),
    dict(inputs=(ts(), ts(dtype="bf16", align=256))),
    dict(inputs=(ts(), ts(off=1, align=4))),
    dict(inputs=(ts(), ts(align=128))),
    dict(outputs=(ts((4, 8), (16, 1)),)),
    dict(attrs=(("alpha_is_one", True),)),
    dict(scalar_kinds=("float",)),
    dict(aliases=((0, 0),)),
    dict(device="NVIDIA GeForce RTX 5060 Ti|sm120|36|33554432"),
]


@pytest.mark.parametrize("kw", VARIANTS)
def test_every_field_changes_identity(kw):
    a, b = cs(), cs(**kw)
    assert a != b and a.digest() != b.digest()


def test_digest_is_stable_across_processes():
    code = (
        "from tests.test_interface import cs;"
        "print(cs(attrs=(('dim', -1), ('eps', 1e-5), ('keepdim', True))).digest())"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env,
                           cwd=os.path.dirname(os.path.dirname(__file__)))
        outs.add(r.stdout.strip())
    assert len(outs) == 1
    assert outs == {cs(attrs=(("dim", -1), ("eps", 1e-5), ("keepdim", True))).digest()}


# --- Param and Impl defaults --------------------------------------------------


def test_param_validation():
    assert Param((128, 256), 256).default == 256
    for bad in [dict(options=()), dict(options=(1, 1)), dict(options=(1, 2), default=3), dict(options=(True,))]:
        with pytest.raises(SpecError):
            Param(**bad)


def test_default_config_from_search_space():
    class I(Impl):
        op = "x"
        search_space = {"bdim": Param((128, 256), 256), "vec": Param((4, 8))}

    assert I().default_config(cs()) == {"bdim": 256, "vec": 4}


# --- registry -----------------------------------------------------------------


def test_registry():
    r = Registry()

    class A(Impl):
        op = "a"

    class A2(Impl):
        op = "a"

    class NoOp(Impl):
        pass

    assert r.register(A) is A and r.get("a") is A and r.ops() == ("a",)
    with pytest.raises(ValueError):
        r.register(A2)
    with pytest.raises(TypeError):
        r.register(NoOp)
    with pytest.raises(TypeError):
        r.register(object)
    with pytest.raises(TypeError):
        r.register(Impl)


def test_register_impl_decorator_uses_global_registry():
    from sol_kernels._runtime.interface import REGISTRY

    @register_impl
    class T(Impl):
        op = "test_interface_only_op"

    assert REGISTRY.get("test_interface_only_op") is T


def test_interface_imports_no_torch_or_dsl():
    code = (
        "import sys; import sol_kernels._runtime.interface;"
        "bad = [m for m in ('torch', 'cutlass', 'cuda') if m in sys.modules]; print(bad)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert r.stdout.strip() == "[]"
