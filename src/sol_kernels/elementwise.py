"""Elementwise apply: out = f(*inputs) per element, with torch's broadcasting.

f takes one Float32 per input and returns a Float32 (or a Boolean for a comparison). As in
torch, each float input is first rounded to the op's common dtype, then widened to Float32,
and the result is rounded to the output dtype. A 0-dim tensor is a scalar operand and
broadcasts like any other input.

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
from math import prod

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import make_ptr

# float64 and integer tensors are left out: f computes in float32. torch.bool is stored
# as one byte, so it is read and written as Uint8.
CUTE_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.bool: cutlass.Uint8,
}
THREADS = 256
VECTOR_BYTES = 16  # the widest global load and store


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


def elementwise_builder(f: Callable, *inputs: torch.Tensor, out_like: torch.Tensor,
                        common_dtype: torch.dtype) -> Callable:
    """Compiles f for the inputs' and the output's shapes, strides and dtypes. out_like
    (a meta tensor is fine) gives the output's shape, strides and dtype; common_dtype is the
    dtype torch rounds the float inputs to before computing. Returns fn(*inputs, out=)."""
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
    outer_strides = [tuple(reversed([s for i, s in enumerate(st) if i != row_dim])) for st in strides]
    rows = prod(outer_shape)

    dtypes = [x.dtype for x in inputs] + [out_like.dtype]
    elem_types = [CUTE_DTYPES[d] for d in dtypes]
    sizes = [torch.empty(0, dtype=d).element_size() for d in dtypes]
    unit_sizes = [size for size, s in zip(sizes, row_strides) if s == 1]  # includes the output
    vec = VECTOR_BYTES // max(unit_sizes) if row_len > 1 else 1
    windows = -(-row_len // vec) + (1 if vec > 1 else 0)  # one more for the peeled head
    assert rows * windows < 2**31  # one thread per window
    out_size = sizes[-1]
    common = CUTE_DTYPES[common_dtype]
    is_bool = [d == torch.bool for d in dtypes[:-1]]

    def to_float(x, k):
        if cutlass.const_expr(is_bool[k]):
            return x.to(Float32)
        return x.to(common).to(Float32)

    @cute.kernel
    def kernel(pIn: tuple, pOut: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        t = bidx * THREADS + tidx
        if t < rows * windows:
            row = t // windows
            # start of this row in every tensor
            starts = []
            for p, st in zip(list(pIn) + [pOut], outer_strides):
                if cutlass.const_expr(rows > 1):
                    starts.append(p + cute.make_layout(outer_shape, stride=st)(row))
                else:
                    starts.append(p)
            pos = (t % windows) * vec
            if cutlass.const_expr(vec > 1):
                # shift the windows so the output's are aligned; the first may start before the row
                pos -= (starts[-1].toint() // out_size) % vec

            if (pos >= 0) & (pos + vec <= row_len):
                rOut = cute.make_rmem_tensor(cute.make_layout(vec), elem_types[-1])
                rIn = []
                for k in cutlass.range_constexpr(len(inputs)):
                    r = cute.make_rmem_tensor(cute.make_layout(vec), elem_types[k])
                    src = starts[k] + pos * row_strides[k]
                    layout = cute.make_layout(vec, stride=row_strides[k])
                    if cutlass.const_expr(row_strides[k] == 1 and vec > 1):
                        if src.toint() % (vec * sizes[k]) == 0:
                            cute.autovec_copy(cute.make_tensor(src.align(vec * sizes[k]), layout), r)
                        else:
                            cute.autovec_copy(cute.make_tensor(src, layout), r)
                    else:
                        cute.autovec_copy(cute.make_tensor(src, layout), r)
                    rIn.append(r)
                for i in cutlass.range_constexpr(vec):
                    rOut[i] = f(*[to_float(r[i], k) for k, r in enumerate(rIn)]).to(elem_types[-1])
                dst = (starts[-1] + pos).align(vec * out_size)
                cute.autovec_copy(rOut, cute.make_tensor(dst, cute.make_layout(vec)))
            else:
                for i in cutlass.range_constexpr(vec):
                    if (pos + i >= 0) & (pos + i < row_len):
                        xs = []
                        for k in cutlass.range_constexpr(len(inputs)):
                            g = cute.make_tensor(starts[k] + (pos + i) * row_strides[k], cute.make_layout(1))
                            xs.append(to_float(g[0], k))
                        out = cute.make_tensor(starts[-1] + (pos + i), cute.make_layout(1))
                        out[0] = f(*xs).to(elem_types[-1])

    @cute.jit
    def launcher(pIn: tuple, pOut: cute.Pointer, stream: cuda.CUstream):
        blocks = (rows * windows + THREADS - 1) // THREADS
        kernel(pIn, pOut).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    def pointers(ts):
        # only element alignment is promised; the kernel checks the rest at run time
        return [make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=s)
                for t, e, s in zip(ts, elem_types, sizes)]

    example = [torch.empty(1, dtype=d, device="cuda" if torch.cuda.is_available() else "cpu") for d in dtypes]
    p = pointers(example)
    compiled = cute.compile(launcher, tuple(p[:-1]), p[-1], cuda.CUstream(0))

    def kernel_wrapper(*xs: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        # The kernel was compiled for these layouts and dtypes and does not check them.
        views = [x.broadcast_to(out_shape) for x in xs]
        assert [v.stride() for v in views] == in_strides
        assert out.shape == out_shape and out.stride() == out_like.stride()
        assert [x.dtype for x in xs] + [out.dtype] == dtypes
        p = pointers(views + [out])
        compiled(tuple(p[:-1]), p[-1], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    return kernel_wrapper
