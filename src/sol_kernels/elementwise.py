"""Elementwise apply: out = f(*inputs) per element, with torch's broadcasting.

As in torch, each input is first converted to the op's common dtype, then to the type the
op computes in, and f's result is converted to the output dtype. The compute type is the
common dtype itself, except that bfloat16 and float16 compute in float32 (torch's
opmath_type). So f takes one value per input in the compute type: Float32 for the 16- and
32-bit floats, Float64, an integer type, or Boolean. Every conversion follows torch's
static_cast (see cast). A 0-dim tensor is a scalar operand and broadcasts like any other
input.

Layout. The inputs are broadcast to the output shape, and neighbouring dims that every
tensor walks as one are merged, so a contiguous 4x3x8 tensor is one run of 96. The kernel
vectorises along the dim where the output has stride 1: a "row" is one line along that dim.

Each row is cut into windows of `vec` elements whose output addresses are 16-byte aligned
(vec * the output's element size bytes, to be exact). One thread handles one window. A
window that lies wholly inside its row stores one vector; an input loads one vector when its
own address there is aligned too, and element by element otherwise, so inputs whose offset
differs from the output's do not slow the output down. The window that holds a row's first
elements before the first aligned address, and the one that holds its last elements, are
cut by the row's ends and run element by element. This peels the unaligned head and tail
inside the same kernel. An input with stride 0 along the row (a broadcast or a scalar) is
loaded once per window; one with any other stride is loaded element by element.
"""

from collections.abc import Callable
from functools import lru_cache
from math import prod

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float16, Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_ptr

# How each dtype is stored. torch.bool is one byte, so it is read and written as Uint8 and
# turned into a Boolean value after loading.
CUTE_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.float64: cutlass.Float64,
    torch.uint8: cutlass.Uint8,
    torch.int8: cutlass.Int8,
    torch.int16: cutlass.Int16,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
    torch.bool: cutlass.Uint8,
}
THREADS = 256
VECTOR_BYTES = 16  # the widest global load and store


def value_type(dtype: torch.dtype):
    """The DSL type of a dtype's values once loaded."""
    return Boolean if dtype == torch.bool else CUTE_DTYPES[dtype]


def compute_type(common_dtype: torch.dtype):
    """The type f computes in: torch computes bfloat16 and float16 in float32."""
    return Float32 if common_dtype in (torch.bfloat16, torch.float16) else value_type(common_dtype)


def cast(x, to):
    """x converted to the DSL type `to` as torch's static_cast_with_inter_type does
    (c10/util/TypeCast.h). c10::Half and c10::BFloat16 are built from a float, so an
    integer or a float64 reaches them through float32, rounding twice. A float reaches
    uint8 through int64, so an out-of-range value wraps instead of saturating."""
    if type(x) is to:
        return x
    if to is Boolean:
        return x != 0
    if to in (Float16, BFloat16) and type(x) not in (Float32, Float16, BFloat16):
        return x.to(Float32).to(to)
    if to is Uint8 and type(x).is_float:
        return x.to(Int64).to(Uint8)
    return x.to(to)


def merge_dims(shape, strides):
    """Drop size-1 dims and merge each pair of neighbouring dims that every tensor walks as
    one. strides holds one stride tuple per tensor. Returns the new shape and strides,
    outermost dim first."""
    dims = [(n, [s[i] for s in strides]) for i, n in enumerate(shape) if n != 1]
    merged = []
    for n, st in dims:
        if merged:
            outer_n, outer_st = merged[-1]
            if all(o == s * n for o, s in zip(outer_st, st)):
                merged[-1] = (outer_n * n, st)
                continue
        merged.append((n, st))
    if not merged:  # every dim had size 1
        merged = [(1, [0] * len(strides))]
    return [n for n, _ in merged], [[st[i] for _, st in merged] for i in range(len(strides))]


@lru_cache(maxsize=None)
def compile_kernel(f: Callable, dtypes: tuple, common_dtype: torch.dtype, row_steps: tuple,
                   n_outer: int, vec: int):
    """The kernel for one variant: f, the tensors' dtypes (the output's last), the common
    dtype, each input's stride along the row where it is static (0 or 1; None for any other
    stride), the number of outer dims (at least one) and the vector width. Shapes, strides and addresses
    are passed at run time, so one compile serves every case of the variant."""
    elem_types = [CUTE_DTYPES[d] for d in dtypes]
    sizes = [torch.empty(0, dtype=d).element_size() for d in dtypes]
    out_size = sizes[-1]
    common, compute = value_type(common_dtype), compute_type(common_dtype)
    out_value = value_type(dtypes[-1])
    n_in = len(dtypes) - 1

    def load(x, k):  # a stored element of input k, in the compute type
        # an element read through a pointer argument has lost its signedness: uint8 reads as int8
        x = cast(x, elem_types[k])
        if cutlass.const_expr(dtypes[k] == torch.bool):
            x = x != 0
        return cast(cast(x, common), compute)

    def store(y):  # f's result, as the output stores it
        return cast(cast(y, out_value), elem_types[-1])

    @cute.kernel
    def kernel(pIn: tuple, pOut: cute.Pointer, row_len: Int32, windows: Int32, n_threads: Int32,
               outer_shape: tuple, outer_strides: tuple, row_strides: tuple):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        t = bidx * THREADS + tidx
        if t < n_threads:
            # the start of this thread's row in every tensor
            row = t // windows
            starts = []
            for k, p in enumerate(list(pIn) + [pOut]):
                starts.append(p + cute.make_layout(outer_shape, stride=outer_strides[k])(row))
            # each tensor's stride along the row, static where it is 0 or 1
            steps = [row_strides[k] if row_steps[k] is None else row_steps[k] for k in range(n_in)]
            pos = (t % windows) * vec
            if cutlass.const_expr(vec > 1):
                # shift the windows so the output's are aligned; the first may start before the row
                pos -= (starts[-1].toint() // out_size) % vec

            if (pos >= 0) & (pos + vec <= row_len):
                rOut = cute.make_rmem_tensor(cute.make_layout(vec), elem_types[-1])
                rIn = []
                for k in cutlass.range_constexpr(n_in):
                    r = cute.make_rmem_tensor(cute.make_layout(vec), elem_types[k])
                    src = starts[k] + pos * steps[k]
                    layout = cute.make_layout(vec, stride=steps[k])
                    if cutlass.const_expr(row_steps[k] == 1 and vec > 1) and src.toint() % (vec * sizes[k]) == 0:
                        cute.autovec_copy(cute.make_tensor(src.align(vec * sizes[k]), layout), r)
                    else:
                        cute.autovec_copy(cute.make_tensor(src, layout), r)
                    rIn.append(r)
                for i in cutlass.range_constexpr(vec):
                    rOut[i] = store(f(*[load(r[i], k) for k, r in enumerate(rIn)]))
                dst = (starts[-1] + pos).align(vec * out_size)
                cute.autovec_copy(rOut, cute.make_tensor(dst, cute.make_layout(vec)))
            else:
                for i in cutlass.range_constexpr(vec):
                    if (pos + i >= 0) & (pos + i < row_len):
                        xs = []
                        for k in cutlass.range_constexpr(n_in):
                            g = cute.make_tensor(starts[k] + (pos + i) * steps[k], cute.make_layout(1))
                            xs.append(load(g[0], k))
                        out = cute.make_tensor(starts[-1] + (pos + i), cute.make_layout(1))
                        out[0] = store(f(*xs))

    @cute.jit
    def launcher(pIn: tuple, pOut: cute.Pointer, row_len: Int32, windows: Int32, n_threads: Int32,
                 outer_shape: tuple, outer_strides: tuple, row_strides: tuple, stream: cuda.CUstream):
        blocks = (n_threads + THREADS - 1) // THREADS
        kernel(pIn, pOut, row_len, windows, n_threads, outer_shape, outer_strides, row_strides).launch(
            grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    # any address will do: the kernel is not specialized on alignment
    p = [make_ptr(e, 4096, cute.AddressSpace.gmem, assumed_align=s) for e, s in zip(elem_types, sizes)]
    one = Int32(1)
    return cute.compile(launcher, tuple(p[:-1]), p[-1], one, one, one, (one,) * n_outer,
                        ((one,) * n_outer,) * len(dtypes), (one,) * n_in, cuda.CUstream(0))


def elementwise_builder(f: Callable, *inputs: torch.Tensor, out_like: torch.Tensor,
                        common_dtype: torch.dtype) -> Callable:
    """The kernel for the inputs' and the output's shapes, strides and dtypes. out_like
    (a meta tensor is fine) gives the output's shape, strides and dtype; common_dtype is the
    dtype torch converts the inputs to before computing. Returns fn(*inputs, out=)."""
    for x in inputs:
        assert x.dtype in CUTE_DTYPES, f"{x.dtype} is not supported"
    assert out_like.dtype in CUTE_DTYPES and common_dtype in CUTE_DTYPES
    out_shape = out_like.shape
    assert torch.broadcast_shapes(*(x.shape for x in inputs)) == out_shape
    assert 0 < out_like.numel() < 2**31  # the kernel indexes with 32-bit integers

    views = [x.broadcast_to(out_shape) for x in inputs]
    in_strides = [v.stride() for v in views]
    shape, strides = merge_dims(out_shape, in_strides + [out_like.stride()])
    for st in strides:  # the furthest element of every tensor is within 32-bit offsets
        assert sum((n - 1) * abs(s) for n, s in zip(shape, st)) < 2**31

    # the row dim: where the output has stride 1. A dense output has one, unless it holds
    # a single element.
    row_dim = strides[-1].index(1) if 1 in strides[-1] else len(shape) - 1
    row_len = shape[row_dim]
    row_strides = [st[row_dim] for st in strides]
    outer_shape = tuple(reversed([n for i, n in enumerate(shape) if i != row_dim]))
    outer_strides = tuple(tuple(reversed([s for i, s in enumerate(st) if i != row_dim])) for st in strides)
    if not outer_shape:  # one row; the kernel still takes one outer dim
        outer_shape, outer_strides = (1,), ((0,),) * len(strides)
    rows = prod(outer_shape)

    dtypes = tuple(x.dtype for x in inputs) + (out_like.dtype,)
    sizes = [torch.empty(0, dtype=d).element_size() for d in dtypes]
    unit_sizes = [size for size, s in zip(sizes, row_strides) if s == 1]  # includes the output
    vec = VECTOR_BYTES // max(unit_sizes) if row_len > 1 else 1
    windows = -(-row_len // vec) + (1 if vec > 1 else 0)  # one more for the peeled head
    assert rows * windows < 2**31  # one thread per window
    row_steps = tuple(s if s in (0, 1) else None for s in row_strides[:-1])
    compiled = compile_kernel(f, dtypes, common_dtype, row_steps, len(outer_shape), vec)

    elem_types = [CUTE_DTYPES[d] for d in dtypes]

    def kernel_wrapper(*xs: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        # The kernel gets these layouts and does not check them against the tensors.
        views = [x.broadcast_to(out_shape) for x in xs]
        assert [v.stride() for v in views] == in_strides
        assert out.shape == out_shape and out.stride() == out_like.stride()
        assert tuple(x.dtype for x in xs) + (out.dtype,) == dtypes
        # only element alignment is promised; the kernel checks the rest at run time
        p = [make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=s)
             for t, e, s in zip(views + [out], elem_types, sizes)]
        compiled(tuple(p[:-1]), p[-1], row_len, windows, rows * windows, outer_shape, outer_strides,
                 tuple(row_strides[:-1]), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    return kernel_wrapper
