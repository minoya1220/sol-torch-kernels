"""CPU tests for sol_compile and the honest copy impl's build path (design 2.3, 6.3).

The CuTe DSL compiles to CUBIN without a GPU, so building is tested here for real,
targeting sm_120a. Running the kernel needs the sm120 run (tests/gpu/).
"""

from __future__ import annotations

import re

import cutlass
import cutlass.cute as cute
import pytest

from sol_kernels._runtime import bridge, compile as rc, dispatch
from sol_kernels._runtime.interface import CallSpec, TensorSpec
from _load import load_copy_honest

DEV = "NVIDIA GeForce RTX 5090|sm120|170|100663296"
copy_mod = load_copy_honest()


def copy_spec(shape=(63, 129), dtype="f32", src_align=256, src_off=0, dst=None, **kw):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.insert(0, acc)
        acc *= s
    src = TensorSpec(tuple(shape), tuple(stride), dtype, src_off, src_align)
    dst = dst or TensorSpec(tuple(shape), tuple(stride), dtype, 0, 256)
    base = dict(op="copy", inputs=(src,), outputs=(dst,), attrs=(), scalar_kinds=(), aliases=(), device=DEV)
    base.update(kw)
    return CallSpec(**base)


def test_arch_from_device():
    assert rc.arch_from_device(DEV) == "sm_120a"
    assert rc.arch_from_device("NVIDIA GeForce RTX 5060 Ti|sm120|36|33554432") == "sm_120a"
    assert rc.arch_from_device("cpu") is None


def test_impls_may_not_pass_compile_options():
    with pytest.raises(ValueError):
        rc.sol_compile(lambda: None, impl="x", options="--link-libraries=/tmp/evil.a")
    with pytest.raises(ValueError):
        rc.sol_compile(lambda: None, impl="x", arch="sm_120a --keep-ptx")


# --- the honest copy impl ------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "ok"),
    [
        (copy_spec(), True),
        (copy_spec(dtype="bf16", src_align=2, src_off=1), True),
        (copy_spec(dtype="bool"), True),
        (copy_spec(shape=(0, 5)), True),
        (copy_spec(shape=()), False),
        (copy_spec(dst=TensorSpec((63, 129), (129, 1), "f16", 0, 256)), False),
        (copy_spec(dst=TensorSpec((129, 63), (1, 129), "f32", 0, 256)), False),
        (copy_spec(dst=TensorSpec((63, 129), (1, 63), "f32", 0, 256)), False),
        (copy_spec(aliases=((0, 0),)), False),
        (copy_spec(attrs=(("x", 1),)), False),
        (copy_spec(shape=(2**31,), dtype="u8"), False),
    ],
)
def test_copy_supports(spec, ok):
    assert copy_mod.CopyHonest().supports(spec) is ok


@pytest.mark.parametrize("spec", [copy_spec(), copy_spec(dtype="bf16", src_align=2, src_off=1), copy_spec(shape=(7,), dtype="i64")],
                         ids=["f32", "bf16-misaligned", "i64"])
def test_copy_builds_for_sm120_on_cpu_and_records_its_kernel(spec):
    rec = rc.CompileRecorder()
    L = dispatch.make_launcher(copy_mod.CopyHonest(), spec, torch_device="cpu", recorder=rec)
    assert callable(L)
    (m,) = rec.modules()
    assert m.arch == "sm_120a" and m.impl.endswith(":CopyHonest")
    (name,) = m.kernel_names
    assert name.startswith("kernel_cutlass_sol_copy_kernel_")
    assert re.fullmatch(r"[0-9a-f]{64}", m.cubin_sha256) and m.ptx_sha256
    assert rec.allowlist() == {name} and rec.module_hashes() == {m.cubin_sha256}


def test_copy_cubin_hash_is_deterministic():
    rec = rc.CompileRecorder()
    for _ in range(2):
        dispatch.make_launcher(copy_mod.CopyHonest(), copy_spec(shape=(33,)), torch_device="cpu", recorder=rec)
    a, b = rec.modules()
    assert a.cubin_sha256 == b.cubin_sha256 and a.kernel_names == b.kernel_names


def test_compile_writes_no_files_outside_a_temp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec = rc.CompileRecorder()
    dispatch.make_launcher(copy_mod.CopyHonest(), copy_spec(shape=(41,)), torch_device="cpu", recorder=rec)
    assert list(tmp_path.iterdir()) == []
    (m,) = rec.modules()
    assert m.cubin_sha256


def test_copy_of_empty_tensor_compiles_nothing():
    rec = rc.CompileRecorder()
    dispatch.make_launcher(copy_mod.CopyHonest(), copy_spec(shape=(0, 5)), torch_device="cpu", recorder=rec)
    assert rec.modules() == ()


# --- recording fails closed -----------------------------------------------------


@cute.kernel
def not_prefixed(src: cute.Tensor, dst: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    dst[tidx] = src[tidx]


@cute.jit
def host_not_prefixed(src: cute.Tensor, dst: cute.Tensor):
    not_prefixed(src, dst).launch(grid=(1, 1, 1), block=(32, 1, 1))


@cute.jit
def host_no_kernel(src: cute.Tensor, dst: cute.Tensor):
    pass


@pytest.mark.parametrize("host", [host_not_prefixed, host_no_kernel], ids=["no-sol-prefix", "no-kernel"])
def test_unrecordable_modules_are_refused(host):
    t = bridge.alloc_with_layout(TensorSpec((32,), (1,), "f32", 0, 256), "cpu")
    a, b = bridge.to_cute(t, bridge.tensor_spec(t)), bridge.to_cute(t, bridge.tensor_spec(t))
    rec = rc.CompileRecorder()
    with pytest.raises(rc.KernelRecordError):
        rc.sol_compile(host, a, b, impl="test", arch="sm_120a", recorder=rec)
    assert rec.modules() == ()


def test_cutlass_int_is_available():
    # guards the copy impl's annotation against a DSL rename
    assert cutlass.Int32 is not None
