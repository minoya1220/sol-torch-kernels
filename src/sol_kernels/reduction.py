"""Batched reduction: reduce a tensor along some of its dims with an op's combine function, and,
for a fused op, run an elementwise pass over each reduced row that uses the row's result
(softmax, layer_norm, rms_norm).

An op (sol_kernels/reduction_ops.py) keeps one state per output, a tuple of Float32 or Int32
parts. `update` folds one element into a state, `combine` merges two states and `finish` turns a
state into the values stored. Threads merge states in whatever order the launch gives, so
combine must be associative and commutative; for argmax that means a tie goes to the lower
index. Each element is widened to Float32 before update, so sums accumulate in float32.

Layout. The input's dims split into kept dims (one output each) and reduced dims. Within each
group, the dims are sorted by the first input's stride and neighbouring dims that every tensor
walks as one are merged, as in the elementwise builder, so a contiguous (A, B, C) reduced over
B and C has one reduced dim of B * C. Merged dims are passed innermost first. Then one of two
modes:

- inner, when the first input has stride 1 along the innermost reduced dim: a team of TX
  threads (a power of two) reduces one output. The innermost reduced dim is the row; any other
  reduced dims are walked one line at a time. Each line is cut into windows of `vec` elements
  aligned in the first input, and the team's threads take the windows in turn. A window inside
  the line loads one vector for each input whose address there is aligned, and element by
  element otherwise; the windows cut by the line's ends run element by element. Each thread
  keeps one state per element of a window, so a float32 sum's serial chains are short, and
  merges them in a tree at the end. The team then merges its states with warp shuffles, and
  through shared memory across warps.
- outer, when the first input has stride 1 along the innermost kept dim instead: each thread
  holds the states of `vec` neighbouring outputs along that dim, in a window aligned like the
  inner mode's, and walks the reduced elements, one vector load per element where the address
  is aligned. The TY threads of a block column share the reduced elements of the same outputs
  and merge through shared memory.

If the first input has stride 1 along no dim, the mode whose fast axis has the smaller stride
runs with vec = 1.

Split. When one block per output team would leave the card idle (a few outputs, each over many
elements), the reduced elements are split across `splits` blocks along grid y. Each block
writes its team's state to a workspace, and a second kernel merges the splits and finishes.

A fused op (one with an `epilogue`) runs in the inner mode with one line per row and no split.
After the team has merged its state, every thread of the team finishes it and writes the
full-size output over the row. The first `cache` windows of each thread stay in registers
from the first pass; the rest are read again, which for long rows mostly hits L2.

Shapes, strides, addresses and the op's scalar parameters are run-time arguments, so one compile
serves every case with the same op, dtypes, mode, team shape, vector width, number of merged
dims, split flag and cache size.
"""

from collections.abc import Callable
from functools import lru_cache
from math import prod

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import make_ptr

from sol_kernels.elementwise import CUTE_DTYPES, VECTOR_BYTES, merge_dims

BLOCK = 256      # threads per block
MAX_CACHE = 8    # windows per thread that a fused op keeps in registers between its two passes
MAX_SPLITS = 64
CPU_SMS = 36  # the RTX 5060 Ti's, so that a compile without a GPU builds the variants that card runs
STATE_DTYPES = {Float32: torch.float32, Int32: torch.int32}


def element_type(dtype: torch.dtype):
    """How a dtype is stored; argmax's int64 indices are the one integer output."""
    return cutlass.Int64 if dtype == torch.int64 else CUTE_DTYPES[dtype]


def pow2_ceil(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def dim_size(shape) -> Int32:
    n = Int32(1)
    for s in shape:
        n = n * s
    return n


@lru_cache(maxsize=None)
def compile_kernels(op, full_dtypes: tuple, row_dtypes: tuple, mode: str, vec: int, tx: int, ty: int,
                    steps: tuple, n_kept: int, n_red: int, split: bool, cache: int, n_params: int):
    """The kernels for one variant. full_dtypes are the dtypes of the full-size tensors: the
    inputs, then a fused op's output; row_dtypes those of the outputs with one element per
    kept coordinate. steps holds each full-size tensor's stride along the vector axis where it
    is static (0 or 1; None for any other stride). Returns the compiled main kernel and, for a
    split, the compiled kernel that merges the splits."""
    full_types = [element_type(d) for d in full_dtypes]
    full_sizes = [torch.empty(0, dtype=d).element_size() for d in full_dtypes]
    row_types = [element_type(d) for d in row_dtypes]
    n_full, n_row, n_parts = len(full_types), len(row_types), len(op.state)
    n_in = n_full - 1 if op.epilogue else n_full
    fused = op.epilogue is not None
    threads = tx * ty

    def load(x, k):
        """An element of input k as the Float32 that update takes; a bool is 0 or 1."""
        if cutlass.const_expr(full_dtypes[k] == torch.bool):
            return cutlass.select_(x != 0, Float32(1.0), Float32(0.0))
        return x.to(Float32)

    @cute.jit
    def new_states(n: cutlass.Constexpr):
        """Registers for n states, one tensor per part, each set to the op's identity."""
        regs = [cute.make_rmem_tensor(cute.make_layout(n), t) for t in op.state]
        for j in cutlass.range_constexpr(n):
            put(regs, j, op.init())
        return regs

    def get(regs, j):
        return tuple(r[j] for r in regs)

    @cute.jit
    def put(regs, j, state):
        for k in cutlass.range_constexpr(n_parts):
            regs[k][j] = state[k]

    @cute.jit
    def store_rows(ptrs, values):
        """Store the first n_row finished values to the row outputs at ptrs."""
        for k in cutlass.range_constexpr(n_row):
            cute.make_tensor(ptrs[k], cute.make_layout(1))[0] = values[k].to(row_types[k])

    def in_line(p, length):
        return (p >= 0) & (p < length)

    def align_shift(p, k):
        """How many elements of tensor k p lies past a vec-aligned address."""
        return Int32((p.toint() // full_sizes[k]) % vec)

    @cute.jit
    def load_window(starts, pos, length, step, n: cutlass.Constexpr):
        """Registers holding the window of vec elements at pos of the first n full-size tensors,
        and whether the window lies wholly inside [0, length). Elements outside stay unset."""
        inside = (pos >= 0) & (pos + vec <= length)
        regs = []
        for k in cutlass.range_constexpr(n):
            r = cute.make_rmem_tensor(cute.make_layout(vec), full_types[k])
            src = starts[k] + pos * step[k]
            if inside:
                aligned = vec * full_sizes[k]
                if cutlass.const_expr(steps[k] == 1 and vec > 1) and src.toint() % aligned == 0:
                    cute.autovec_copy(cute.make_tensor(src.align(aligned), cute.make_layout(vec)), r)
                else:
                    cute.autovec_copy(cute.make_tensor(src, cute.make_layout(vec, stride=step[k])), r)
            else:
                for i in cutlass.range_constexpr(vec):
                    if in_line(pos + i, length):
                        r[i] = cute.make_tensor(starts[k] + (pos + i) * step[k], cute.make_layout(1))[0]
            regs.append(r)
        return regs, inside

    @cute.jit
    def fold_window(states, at, pos, row_len, step, index, n: cutlass.Constexpr):
        """Load the window at pos of the first n full-size tensors and fold element i of it, if
        inside the row, into state i. Returns the window's registers."""
        regs, inside = load_window(at, pos, row_len, step, n)
        for i in cutlass.range_constexpr(vec):
            if inside | in_line(pos + i, row_len):
                fold(states, i, regs, i, index + (pos + i) * step[n_full])
        return regs

    @cute.jit
    def fold(states, j, regs, i, index):
        xs = tuple(load(regs[k][i], k) for k in range(op.reduce_inputs))
        put(states, j, op.update(get(states, j), xs, index))

    @cute.jit
    def merge_vec(states):
        """Merge the vec states into state 0, in a tree."""
        for b in cutlass.range_constexpr(vec.bit_length() - 1):
            h = vec >> (b + 1)
            for j in cutlass.range_constexpr(h):
                put(states, j, op.combine(get(states, j), get(states, j + h)))

    @cute.jit
    def merge_lanes(states, width: cutlass.Constexpr):
        """Merge state 0 across groups of `width` neighbouring lanes; every lane gets the result."""
        for b in cutlass.range_constexpr(width.bit_length() - 1):
            s = get(states, 0)
            put(states, 0, op.combine(s, tuple(cute.arch.shuffle_sync_bfly(v, 1 << b) for v in s)))

    @cute.kernel
    def inner_kernel(p_full: tuple, p_row: tuple, p_work: tuple, kept_shape: tuple, kept_strides: tuple,
                     red_shape: tuple, red_strides: tuple, count: Float32, params: tuple):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, by, _ = cute.arch.block_idx()
        _, splits, _ = cute.arch.grid_dim()
        lane, c = tidx % tx, bidx * (BLOCK // tx) + tidx // tx  # c: the output this thread's team reduces
        lane, team = by * tx + lane, splits * tx
        n_out = dim_size(kept_shape)
        row_len, n_lines = red_shape[0], dim_size(red_shape[1:])
        # each tensor's stride along the row, static where it is 0 or 1
        step = [red_strides[k][0] if steps[k] is None else steps[k] for k in range(n_full)] + [red_strides[n_full][0]]
        starts = [p + cute.make_layout(kept_shape, stride=kept_strides[k])(c) for k, p in enumerate(p_full)]
        lines = [cute.make_layout(red_shape, stride=red_strides[k]) for k in range(n_full + 1)]  # the last gives indices

        # one state per element of a window, so that a sum's serial chains are vec times shorter
        states = new_states(vec)
        cached = [cute.make_rmem_tensor(cute.make_layout((vec, max(cache, 1))), full_types[k]) for k in range(n_in)]
        if c < n_out:
            for line in range(n_lines):
                at = [s + lines[k](line * row_len) for k, s in enumerate(starts)]
                index = lines[-1](line * row_len)
                shift = align_shift(at[0], 0)
                n_win = (row_len + shift + vec - 1) // vec
                for j in cutlass.range_constexpr(cache):  # only a fused op caches; it has one line
                    if lane + j * team < n_win:
                        regs_c = fold_window(states, at, (lane + j * team) * vec - shift, row_len, step, index, n_in)
                        for k in cutlass.range_constexpr(n_in):
                            for i in cutlass.range_constexpr(vec):
                                cached[k][i, j] = regs_c[k][i]
                for w in range(lane + cache * team, n_win, team):
                    fold_window(states, at, w * vec - shift, row_len, step, index, op.reduce_inputs)

        # merge the thread's states pairwise, then the team: across lanes in a warp, then across
        # the team's warps
        merge_vec(states)
        merge_lanes(states, min(tx, 32))
        if cutlass.const_expr(tx > 32):
            smem = cutlass.memory.SmemAllocator()
            buffers = [smem.allocate_tensor(t, cute.make_layout(BLOCK // 32), byte_alignment=16) for t in op.state]
            if tidx % 32 == 0:
                for k in cutlass.range_constexpr(n_parts):
                    buffers[k][tidx // 32] = states[k][0]
            cute.arch.sync_threads()
            warps = tx // 32
            for k in cutlass.range_constexpr(n_parts):
                states[k][0] = buffers[k][(tidx // tx) * warps + tidx % warps]
            merge_lanes(states, warps)

        if c < n_out:
            if cutlass.const_expr(split):
                if tidx % tx == 0:
                    for k in cutlass.range_constexpr(n_parts):
                        cute.make_tensor(p_work[k] + by * n_out + c, cute.make_layout(1))[0] = states[k][0]
            else:
                values = op.finish(get(states, 0), count, params)
                if tidx % tx == 0:
                    store_rows([p + cute.make_layout(kept_shape, stride=kept_strides[n_full + k])(c)
                                for k, p in enumerate(p_row[:n_row])], values)
                if cutlass.const_expr(fused):
                    shift = align_shift(starts[0], 0)
                    write_row(starts, values, lane, team, row_len, shift, step, cached, params)

    @cute.jit
    def write(starts, regs, values, pos, row_len, step, params):
        """A fused op's output over one window: y = epilogue(xs, values, params) per element."""
        inside = (pos >= 0) & (pos + vec <= row_len)
        y = cute.make_rmem_tensor(cute.make_layout(vec), full_types[n_full - 1])
        for i in cutlass.range_constexpr(vec):
            xs = tuple(load(regs[k][i], k) for k in range(n_in))
            y[i] = op.epilogue(xs, values, params).to(full_types[n_full - 1])
        dst = starts[n_full - 1] + pos * step[n_full - 1]
        if inside:
            aligned = vec * full_sizes[n_full - 1]
            if cutlass.const_expr(steps[n_full - 1] == 1 and vec > 1) and dst.toint() % aligned == 0:
                cute.autovec_copy(y, cute.make_tensor(dst.align(aligned), cute.make_layout(vec)))
            else:
                cute.autovec_copy(y, cute.make_tensor(dst, cute.make_layout(vec, stride=step[n_full - 1])))
        else:
            for i in cutlass.range_constexpr(vec):
                if in_line(pos + i, row_len):
                    cute.make_tensor(starts[n_full - 1] + (pos + i) * step[n_full - 1], cute.make_layout(1))[0] = y[i]

    @cute.jit
    def write_row(starts, values, lane, team, row_len, shift, step, cached, params):
        """A fused op's output over the row, from the cached windows and then the rest."""
        n_win = (row_len + shift + vec - 1) // vec
        for j in cutlass.range_constexpr(cache):
            if lane + j * team < n_win:
                regs_c = [cute.make_rmem_tensor(cute.make_layout(vec), full_types[k]) for k in range(n_in)]
                for k in cutlass.range_constexpr(n_in):
                    for i in cutlass.range_constexpr(vec):
                        regs_c[k][i] = cached[k][i, j]
                write(starts, regs_c, values, (lane + j * team) * vec - shift, row_len, step, params)
        for w in range(lane + cache * team, n_win, team):
            regs, _ = load_window(starts, w * vec - shift, row_len, step, n_in)
            write(starts, regs, values, w * vec - shift, row_len, step, params)

    @cute.kernel
    def outer_kernel(p_full: tuple, p_row: tuple, p_work: tuple, kept_shape: tuple, kept_strides: tuple,
                     red_shape: tuple, red_strides: tuple, count: Float32, params: tuple):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, by, _ = cute.arch.block_idx()
        _, splits, _ = cute.arch.grid_dim()
        col, lane = tidx % tx, tidx // tx
        lane, team = by * ty + lane, splits * ty
        # the kept dims as lines along the innermost kept dim, each cut into windows
        kept_len, n_lines = kept_shape[0], dim_size(kept_shape[1:])
        n_win = (kept_len + vec - 1) // vec + (1 if vec > 1 else 0)
        t = bidx * tx + col
        line = t // n_win
        step = [kept_strides[k][0] if steps[k] is None else steps[k] for k in range(n_full)]
        starts = [p + cute.make_layout(kept_shape, stride=kept_strides[k])(line * kept_len) for k, p in enumerate(p_full)]
        pos = (t % n_win) * vec
        if cutlass.const_expr(vec > 1):
            pos -= align_shift(starts[0], 0)
        reduced = [cute.make_layout(red_shape, stride=red_strides[k]) for k in range(n_full + 1)]

        states = new_states(vec)
        if line < n_lines:
            for r in range(lane, dim_size(red_shape), team):
                regs, inside = load_window([s + reduced[k](r) for k, s in enumerate(starts)], pos, kept_len, step,
                                           op.reduce_inputs)
                for i in cutlass.range_constexpr(vec):
                    if inside | in_line(pos + i, kept_len):
                        fold(states, i, regs, i, reduced[-1](r))

        # merge the TY threads that share these outputs
        smem = cutlass.memory.SmemAllocator()
        buffers = [smem.allocate_tensor(t, cute.make_layout((vec, threads)), byte_alignment=16) for t in op.state]
        for k in cutlass.range_constexpr(n_parts):
            for i in cutlass.range_constexpr(vec):
                buffers[k][i, tidx] = states[k][i]
        cute.arch.sync_threads()
        if (tidx < tx) & (line < n_lines):
            for y in cutlass.range_constexpr(1, ty):
                for i in cutlass.range_constexpr(vec):
                    put(states, i, op.combine(get(states, i), tuple(b[i, y * tx + col] for b in buffers)))
            for i in cutlass.range_constexpr(vec):
                if in_line(pos + i, kept_len):
                    if cutlass.const_expr(split):
                        c = line * kept_len + pos + i
                        for k in cutlass.range_constexpr(n_parts):
                            cute.make_tensor(p_work[k] + by * n_lines * kept_len + c, cute.make_layout(1))[0] = states[k][i]
                    else:
                        store_rows([p + cute.make_layout(kept_shape, stride=kept_strides[n_full + k])(line * kept_len)
                                    + (pos + i) * kept_strides[n_full + k][0] for k, p in enumerate(p_row[:n_row])],
                                   op.finish(get(states, i), count, params))

    @cute.kernel
    def merge_kernel(p_row: tuple, p_work: tuple, kept_shape: tuple, kept_strides: tuple, count: Float32,
                     params: tuple, splits: Int32):
        """Merge the splits' states of each output, finish and store."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        c = bidx * BLOCK + tidx
        n_out = dim_size(kept_shape)
        if c < n_out:
            states = new_states(1)
            for s in range(splits):
                part = tuple(cute.make_tensor(p + s * n_out + c, cute.make_layout(1))[0] for p in p_work)
                put(states, 0, op.combine(get(states, 0), part))
            store_rows([p + cute.make_layout(kept_shape, stride=kept_strides[n_full + k])(c) for k, p in enumerate(p_row[:n_row])],
                       op.finish(get(states, 0), count, params))

    @cute.jit
    def launch(p_full: tuple, p_row: tuple, p_work: tuple, kept_shape: tuple, kept_strides: tuple,
               red_shape: tuple, red_strides: tuple, count: Float32, params: tuple, splits: Int32,
               stream: cuda.CUstream):
        args = (p_full, p_row, p_work, kept_shape, kept_strides, red_shape, red_strides, count, params)
        n_out = dim_size(kept_shape)
        if cutlass.const_expr(mode == "inner"):
            blocks = (n_out + BLOCK // tx - 1) // (BLOCK // tx)
            inner_kernel(*args).launch(grid=(blocks, splits, 1), block=(BLOCK, 1, 1), stream=stream)
        else:
            n_win = (kept_shape[0] + vec - 1) // vec + (1 if vec > 1 else 0)
            blocks = (dim_size(kept_shape[1:]) * n_win + tx - 1) // tx
            outer_kernel(*args).launch(grid=(blocks, splits, 1), block=(threads, 1, 1), stream=stream)
        if cutlass.const_expr(split):
            merge_kernel(p_row, p_work, kept_shape, kept_strides, count, params, splits).launch(
                grid=((n_out + BLOCK - 1) // BLOCK, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    # placeholders to compile with: any address will do, as the kernels are not specialized on alignment
    def ptrs(types):
        return tuple(make_ptr(t, 4096, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in types)

    one = Int32(1)
    kept, red = (one,) * n_kept, (one,) * n_red
    # a tuple argument cannot be empty, so softmax gets one unused row pointer, an op without
    # a split one workspace pointer per state part, and one without parameters a parameter of 0
    return cute.compile(launch, ptrs(full_types), ptrs(row_types or [Float32]), ptrs(op.state), kept,
                        (kept,) * (n_full + n_row), red, (red,) * (n_full + 1), Float32(1.0),
                        (Float32(0.0),) * max(n_params, 1), one, cuda.CUstream(0))


def reduction_builder(op, *inputs: torch.Tensor, dims: tuple, outs_like: tuple, params: tuple = ()) -> Callable:
    """The kernel that reduces inputs[0] (and the op's other reduction inputs) over dims, a
    tuple of its dims. Every input broadcasts to inputs[0]'s shape. outs_like (meta tensors are
    fine) gives each output's shape, strides and dtype in torch's order: a fused op's full-size
    output first, then the outputs with one element per kept coordinate, with or without
    keepdim. params are the op's scalar parameters. Returns fn(*inputs, out=), where out is the
    tuple of outputs."""
    x = inputs[0]
    shape = tuple(x.shape)
    fused = op.epilogue is not None
    full_like = [t.broadcast_to(shape) for t in inputs] + list(outs_like[:1] if fused else [])
    row_like = list(outs_like[1:] if fused else outs_like)
    for t in full_like + row_like:
        assert t.dtype in CUTE_DTYPES or t.dtype == torch.int64, f"{t.dtype} is not supported"
    assert tuple(full_like[-1].shape) == shape
    assert x.numel() < 2**31  # the kernels index with 32-bit integers
    kept_dims = [d for d in range(len(shape)) if d not in dims]

    def row_strides(o):
        """A row output's stride along each of x's dims: 0 along the reduced ones."""
        st = iter(o.stride())
        if o.dim() < len(shape):  # without keepdim, the reduced dims are gone
            return [0 if d in dims else next(st) for d in range(len(shape))]
        return [0 if d in dims else s for d, s in zip(range(len(shape)), st)]

    for o in row_like:  # without keepdim, or with it
        assert list(o.shape) in ([n for d, n in enumerate(shape) if d not in dims], [1 if d in dims else n for d, n in enumerate(shape)])
    full_st = [t.stride() for t in full_like]
    row_st = [row_strides(o) for o in row_like]
    # an element's index, as torch's argmax counts it: row-major over the reduced dims alone
    index_st = [prod(shape[e] for e in dims if e > d) if d in dims else 0 for d in range(len(shape))]

    def group(ds, strides):
        """Merged dims and strides of the dims ds, innermost first."""
        ds = sorted(ds, key=lambda d: -abs(full_st[0][d]))
        n, st = merge_dims([shape[d] for d in ds], [[s[d] for d in ds] for s in strides])
        return tuple(reversed(n)), tuple(tuple(reversed(s)) for s in st)

    kept_n, kept_st = group(kept_dims, full_st + row_st)
    red_n, red_st = group(dims, full_st + [index_st])
    n_out, n_red = prod(kept_n), prod(red_n)

    # the mode: which axis the threads walk with vector loads
    if fused or red_st[0][0] == 1 or (kept_st[0][0] != 1 and abs(red_st[0][0]) <= abs(kept_st[0][0])):
        mode, axis = "inner", [s[0] for s in red_st[:len(full_like)]]
        axis_len = red_n[0]
    else:
        mode, axis = "outer", [s[0] for s in kept_st[:len(full_like)]]
        axis_len = kept_n[0]
    sizes = [t.element_size() for t in full_like]
    vec = VECTOR_BYTES // max(size for size, s in zip(sizes, axis) if s == 1) if axis[0] == 1 and axis_len > 1 else 1
    steps = tuple(s if s in (0, 1) else None for s in axis)

    # the team shape, and whether to split the reduced elements across blocks
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count if x.is_cuda else CPU_SMS
    target = 4 * sms  # blocks that keep the card busy
    windows = -(-axis_len // vec)
    if mode == "inner":
        tx = min(BLOCK, pow2_ceil(windows))
        ty = BLOCK // tx
        blocks = -(-n_out // ty)
        splits = min(-(-target // max(blocks, 1)), windows // (4 * tx))
    else:
        n_win = windows + (1 if vec > 1 else 0)
        tx = min(32, pow2_ceil(n_out // max(axis_len, 1) * n_win))
        ty = min(BLOCK // tx, pow2_ceil(n_red))
        blocks = -(-(n_out // max(axis_len, 1) * n_win) // tx)
        splits = min(-(-target // max(blocks, 1)), n_red // (4 * ty))
    splits = 1 if fused else max(1, min(splits, MAX_SPLITS))
    cache = min(MAX_CACHE, -(-windows // tx)) if fused else 0
    work = [torch.empty(max(splits * n_out, 1), dtype=STATE_DTYPES[t], device=x.device) for t in op.state]
    for st in kept_st + red_st:  # the furthest element of every tensor is within 32-bit offsets
        assert sum((n - 1) * abs(s) for n, s in zip(kept_n + red_n, st)) < 2**31

    compiled = compile_kernels(op, tuple(t.dtype for t in full_like), tuple(o.dtype for o in row_like), mode, vec,
                               tx, ty, steps, len(kept_n), len(red_n), splits > 1, cache, len(params))
    full_types = [element_type(t.dtype) for t in full_like]
    row_types = [element_type(o.dtype) for o in row_like]

    def kernel_wrapper(*xs: torch.Tensor, out: tuple) -> tuple:
        # The kernels get these layouts and do not check them against the tensors.
        outs = out if isinstance(out, tuple) else (out,)
        views = [t.broadcast_to(shape) for t in xs] + list(outs[:1] if fused else [])
        rows = outs[1:] if fused else outs
        assert [t.stride() for t in views] == full_st and [t.dtype for t in views] == [t.dtype for t in full_like]
        assert [(o.shape, o.stride(), o.dtype) for o in rows] == [(o.shape, o.stride(), o.dtype) for o in row_like]
        if n_out == 0:
            return out
        # only element alignment is promised; the kernels check the rest at run time
        p_full = tuple(make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=t.element_size())
                       for t, e in zip(views, full_types))
        p_row = tuple(make_ptr(e, o.data_ptr(), cute.AddressSpace.gmem, assumed_align=o.element_size())
                      for o, e in zip(rows or work[:1], row_types or [Float32]))
        p_work = tuple(make_ptr(e, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=4) for w, e in zip(work, op.state))
        compiled(p_full, p_row, p_work, kept_n, kept_st, red_n, red_st, float(n_red), tuple(float(p) for p in params or (0.0,)),
                 splits, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    return kernel_wrapper
