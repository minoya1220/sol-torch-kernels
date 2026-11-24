"""Elementwise apply: out = f(*inputs) per element, with torch's broadcasting.

f takes one Float32 per input and returns a Float32 (or a Boolean for a comparison). Every
input is converted to Float32 before f and the result to the output dtype after it, as
torch does for 16-bit floats. A 0-dim tensor is a scalar operand; it broadcasts like any
other input.

The builder broadcasts the inputs to the output shape and merges dims that every tensor
walks with the same stride pattern, so contiguous inputs become one flat dim. Each thread
handles one vector of `vec` elements along the innermost dim. `vec` is as wide as all the
tensors allow (16-byte loads at most) and falls to 1 for a layout that does not allow
vectors, which is the slow but correct path for strided and unaligned inputs.
"""

from collections.abc import Callable
from math import gcd

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32
from cutlass.cute.runtime import make_ptr

from sol_kernels.cute_utils import pointer_alignment

# float64 and integer tensors are left out: f computes in float32. torch.bool is stored
# as one byte, so it is read and written as Uint8.
CUTE_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.bool: cutlass.Uint8,
}
THREADS = 256


def merge_dims(shape, strides):
    """Drop size-1 dims and merge each pair of neighbouring dims that every tensor walks as
    one, so that a contiguous tensor becomes one dim. strides holds one stride tuple per
    tensor. Returns the new shape and strides, outermost dim first."""
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


def vector_width(shape, strides, tensors) -> int:
    """Elements per thread: the widest vector, up to 16 bytes, that every tensor which is
    contiguous along the innermost dim can load from every row start, given its real
    pointer alignment. 1 when some tensor is strided along the innermost dim."""
    if any(st[-1] not in (0, 1) for st in strides):
        return 1
    vec = shape[-1]
    for t, st in zip(tensors, strides):
        if st[-1] == 1:
            vec = gcd(vec, pointer_alignment(t) // t.element_size())
            for s in st[:-1]:
                vec = gcd(vec, s)
    return vec


def elementwise_builder(f: Callable, *inputs: torch.Tensor, out_dtype: torch.dtype) -> Callable:
    """Compiles f for the inputs' shapes, strides, dtypes and alignment. Returns
    fn(*inputs, out=), where out is a contiguous tensor of the broadcast shape and out_dtype."""
    for x in inputs:
        assert x.dtype in CUTE_DTYPES, f"{x.dtype} is not supported"
    assert out_dtype in CUTE_DTYPES

    out_shape = torch.broadcast_shapes(*(x.shape for x in inputs))
    out = torch.empty(out_shape, dtype=out_dtype, device=inputs[0].device)
    tensors = [x.broadcast_to(out_shape) for x in inputs] + [out]
    numel = out.numel()
    assert 0 < numel < 2**31  # the kernel indexes with 32-bit integers

    shape, strides = merge_dims(out_shape, [t.stride() for t in tensors])
    vec = vector_width(shape, strides, tensors)
    aligns = [pointer_alignment(t) for t in tensors]
    elem_types = [CUTE_DTYPES[t.dtype] for t in tensors]
    n_vectors = numel // vec

    # CuTe walks a layout's first mode fastest, so the modes are listed innermost first.
    # The innermost dim is split into (vec, rest): mode 0 of a tiled view is one vector.
    tile_shape = (vec, (shape[-1] // vec, *reversed(shape[:-1])))
    tile_strides = [(st[-1], (st[-1] * vec, *reversed(st[:-1]))) for st in strides]

    @cute.kernel
    def kernel(gIn: tuple, gOut: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        v = bidx * THREADS + tidx
        if v < n_vectors:
            rIn = []
            for g in gIn:
                r = cute.make_rmem_tensor(cute.make_layout(vec), g.element_type)
                cute.autovec_copy(g[None, v], r)
                rIn.append(r)
            rOut = cute.make_rmem_tensor(cute.make_layout(vec), gOut.element_type)
            for i in cutlass.range_constexpr(vec):
                y = f(*[r[i].to(Float32) for r in rIn])
                rOut[i] = y.to(gOut.element_type)
            cute.autovec_copy(rOut, gOut[None, v])

    @cute.jit
    def launcher(pIn: tuple, pOut: cute.Pointer, stream: cuda.CUstream):
        gIn = tuple(cute.make_tensor(p, cute.make_layout(tile_shape, stride=st)) for p, st in zip(pIn, tile_strides))
        gOut = cute.make_tensor(pOut, cute.make_layout(tile_shape, stride=tile_strides[-1]))
        blocks = (n_vectors + THREADS - 1) // THREADS
        kernel(gIn, gOut).launch(grid=(blocks, 1, 1), block=(THREADS, 1, 1), stream=stream)

    def pointers(ts):
        return [make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=a)
                for t, e, a in zip(ts, elem_types, aligns)]

    ptrs = pointers(tensors)
    compiled = cute.compile(launcher, tuple(ptrs[:-1]), ptrs[-1], cuda.CUstream(0))
    in_strides = [t.stride() for t in tensors[:-1]]

    def kernel_wrapper(*xs: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        # The kernel was compiled for one layout, dtype and alignment and does not check them.
        views = [x.broadcast_to(out_shape) for x in xs]
        assert [v.stride() for v in views] == in_strides and out.shape == out_shape and out.is_contiguous()
        assert [x.dtype for x in xs] == [t.dtype for t in tensors[:-1]] and out.dtype == out_dtype
        assert all(pointer_alignment(t) >= a for t, a in zip(views + [out], aligns))
        p = pointers(views + [out])
        compiled(tuple(p[:-1]), p[-1], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    return kernel_wrapper
