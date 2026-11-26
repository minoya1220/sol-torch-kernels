"""Attention forward: torch.nn.functional.scaled_dot_product_attention with no mask other
than causal and no dropout. A flash-attention-2 kernel, ported from the CUTLASS example
examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py (v4.8.1).

sm120 has no wgmma (sm90 only) and no tcgen05 or tensor memory (sm100 only); its tensor
cores take the warp-level mma.sync instructions of sm80, so the Ampere design is the one
that fits. Each CTA takes a block of query rows of one head and walks the keys in blocks: S = Q K^T
on the tensor cores in float32, an online softmax in registers, P rounded to the input dtype,
then O += P V in float32. cp.async loads the next K block while P V runs and the next V block
while Q K^T runs.

Changes from the example:
- tensors are torch's (batch, heads, seq, head_dim), contiguous. Batch, heads and sequence
  lengths are run-time values, so one compile serves every shape of a dtype, head dim and
  causal flag;
- grouped-query attention: query head h reads key and value head h // (heads // kv_heads);
- the causal mask is torch's: query i sees keys 0..i, also when Sq != Sk;
- a row's running max stays -inf until the row has seen a key. The example set it to 0, so
  when a causal row's later keys all had scaled scores below about -87, exp2 underflowed to 0
  on every one of them and the row came out as zeros;
- query x key tiles of 128 x 64 for head dim 64 and 64 x 64 for 128: with 128 query rows at
  head dim 128, ptxas spilled about 150 bytes per thread;
- the CTAs of a causal kernel start with the longest rows, so short ones fill the tail;
- the log-sum-exp of each row is written as well, for the backward.
"""

import math
from collections.abc import Callable
from functools import lru_cache
from types import SimpleNamespace

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_ptr

CUTE_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}
THREADS = 128  # 4 warps; with the 16 x 8 x 16 mma, each warp owns 16 of every 64 query rows
LOG2_E = math.log2(math.e)


@lru_cache(maxsize=None)
def compile_kernel(dtype: torch.dtype, head_dim: int, causal: bool):
    """The kernel for one dtype, head dim and causal flag. Shapes and the scale are passed at
    run time."""
    elem = CUTE_DTYPES[dtype]
    D = head_dim
    # Query rows per CTA and keys per block, the largest that keep the accumulators in
    # registers without spilling (ptxas -v, CuTe DSL 4.8.1). Q, K and V take 32 KiB of shared
    # memory for D = 64 and 48 KiB for D = 128, so two CTAs fit on an SM.
    M_BLOCK = 128 if D == 64 else 64
    N_BLOCK = 64
    mask_steps = M_BLOCK // N_BLOCK if causal else 1

    def mn_view(acc: cute.Tensor) -> cute.Tensor:
        """An accumulator fragment ((2, 2), MMA_M, MMA_N) viewed as (rows, columns) of this thread."""
        col_major = cute.make_layout(acc.layout.shape)
        mn = cute.make_layout(
            ((col_major.shape[0][1], col_major.shape[1]), (col_major.shape[0][0], col_major.shape[2])),
            stride=((col_major.stride[0][1], col_major.stride[1]), (col_major.stride[0][0], col_major.stride[2])),
        )
        return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn))

    def quad_reduce(val, op):
        """Reduce across the 4 threads that hold one row of an mma accumulator."""
        val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
        return op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))

    @cute.jit
    def online_softmax(s: SimpleNamespace, acc_S: cute.Tensor, n_block: Int32,
                       is_first: cutlass.Constexpr, masked: cutlass.Constexpr):
        """Mask S, update each row's running max and sum, rescale O, and leave exp(S - max) in acc_S."""
        acc_S_mn = mn_view(acc_S)
        acc_O_mn = mn_view(s.acc_O)
        tScS_mn = None
        if cutlass.const_expr(masked):
            cS = cute.local_tile(cute.make_identity_tensor((s.Sq, s.Sk)), (M_BLOCK, N_BLOCK), (s.m_block, n_block))
            tScS_mn = mn_view(s.thr_mma.partition_C(cS))

        for r in cutlass.range_constexpr(cute.size(s.row_max)):
            if cutlass.const_expr(masked):
                col_limit = s.Sk  # keys past the end
                if cutlass.const_expr(causal):
                    col_limit = cutlass.min(tScS_mn[r, 0][0] + 1, s.Sk)
                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    if tScS_mn[0, c][1] >= col_limit:
                        acc_S_mn[r, c] = -Float32.inf

            row = acc_S_mn[r, None].load()
            row_max = quad_reduce(row.reduce(cute.ReductionOp.MAX, -Float32.inf, 0), cute.arch.fmax)
            if cutlass.const_expr(not is_first):
                row_max = cute.arch.fmax(s.row_max[r], row_max)
            # A row that has seen no key yet keeps max -inf; shift it by 0, so that exp2 of its
            # masked scores is 0 rather than NaN.
            shift = 0.0 if row_max == -Float32.inf else row_max * s.scale_log2
            p = cute.math.exp2(row * s.scale_log2 - shift, fastmath=True)
            row_sum = p.reduce(cute.ReductionOp.ADD, Float32.zero, 0)
            if cutlass.const_expr(not is_first):
                # 0 while the row had seen no key: its old max is -inf
                correction = cute.math.exp2(s.row_max[r] * s.scale_log2 - shift, fastmath=True)
                row_sum = row_sum + s.row_sum[r] * correction
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * correction
            s.row_max[r] = row_max
            s.row_sum[r] = row_sum
            acc_S_mn[r, None] = p

    @cute.jit
    def attend_block(s: SimpleNamespace, n_block: Int32, is_first: cutlass.Constexpr, masked: cutlass.Constexpr):
        """O += softmax-weighted V for one block of keys, with K already in flight to smem."""
        acc_S = cute.make_rmem_tensor(s.thr_mma.partition_shape_C((M_BLOCK, N_BLOCK)), Float32)
        acc_S.fill(0.0)

        # wait for K, then start loading V while Q K^T runs
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if cutlass.const_expr(is_first):
            # the first block may run past Sk: zero those rows, so that P V reads no garbage
            for n in cutlass.range_constexpr(cute.size(s.tVsV.shape[1])):
                if s.tKVcKV[0, n, 0][0] < s.Sk:
                    cute.copy(s.gmem_copy, s.tVgV[None, n, None, n_block], s.tVsV[None, n, None])
                else:
                    s.tVsV[None, n, None].fill(0.0)
        else:
            cute.copy(s.gmem_copy, s.tVgV[None, None, None, n_block], s.tVsV)
        cute.arch.cp_async_commit_group()

        # S = Q K^T, loading the next k-slice of Q and K from smem while the mma runs
        n_k = cute.size(s.tSsQ.shape[2])
        cute.copy(s.smem_copy_Q, s.tSsQ[None, None, 0], s.tSrQ_view[None, None, 0])
        cute.copy(s.smem_copy_K, s.tSsK[None, None, 0], s.tSrK_view[None, None, 0])
        for k in cutlass.range_constexpr(n_k):
            k_next = (k + 1) % n_k
            cute.copy(s.smem_copy_Q, s.tSsQ[None, None, k_next], s.tSrQ_view[None, None, k_next])
            cute.copy(s.smem_copy_K, s.tSsK[None, None, k_next], s.tSrK_view[None, None, k_next])
            cute.gemm(s.tiled_mma, acc_S, s.tSrQ[None, None, k], s.tSrK[None, None, k], acc_S)

        # wait for V, then start loading the next K block (the blocks run from last to first)
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        if n_block > 0:
            cute.copy(s.gmem_copy, s.tKgK[None, None, None, n_block - 1], s.tKsK)
            cute.arch.cp_async_commit_group()

        online_softmax(s, acc_S, n_block, is_first, masked)

        # P in the input dtype, viewed as the A operand of the P V mma: the accumulator's
        # (4, MMA_M, MMA_N) becomes ((4, 2), MMA_M, MMA_N / 2), since one 16 x 16 A tile
        # spans two 16 x 8 accumulator tiles
        rP = cute.make_fragment_like(acc_S, elem)
        rP.store(acc_S.load().to(elem))
        divided = cute.logical_divide(rP.layout, (None, None, 2))
        tOrP = cute.make_tensor(rP.iterator, cute.make_layout(
            ((divided.shape[0], divided.shape[2][0]), divided.shape[1], divided.shape[2][1]),
            stride=((divided.stride[0], divided.stride[2][0]), divided.stride[1], divided.stride[2][1]),
        ))

        # O += P V
        n_k = cute.size(tOrP.shape[2])
        cute.copy(s.smem_copy_V, s.tOsVt[None, None, 0], s.tOrVt_view[None, None, 0])
        for k in cutlass.range_constexpr(n_k):
            k_next = (k + 1) % n_k
            cute.copy(s.smem_copy_V, s.tOsVt[None, None, k_next], s.tOrVt_view[None, None, k_next])
            cute.gemm(s.tiled_mma, s.acc_O, tOrP[None, None, k], s.tOrVt[None, None, k], s.acc_O)

    @cute.kernel
    def kernel(mQ: cute.Tensor, mK: cute.Tensor, mV: cute.Tensor, mO: cute.Tensor, mLSE: cute.Tensor,
               scale_log2: Float32, sQ_layout: cute.ComposedLayout, sKV_layout: cute.ComposedLayout,
               gmem_copy: cute.TiledCopy, gmem_copy_O: cute.TiledCopy, tiled_mma: cute.TiledMma,
               SharedStorage: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        m_block, head, batch = cute.arch.block_idx()
        if cutlass.const_expr(causal):
            m_block = cute.arch.grid_dim()[0] - 1 - m_block  # the longest rows first
        kv_head = head // (mQ.shape[1] // mK.shape[1])
        Sq, Sk = mQ.shape[2], mK.shape[2]

        n_blocks = cute.ceil_div(Sk, N_BLOCK)
        if cutlass.const_expr(causal):  # the keys of the last row in this CTA, and no more
            n_blocks = cutlass.min(cute.ceil_div((m_block + 1) * M_BLOCK, N_BLOCK), n_blocks)

        gQ = cute.local_tile(mQ[batch, head, None, None], (M_BLOCK, D), (m_block, 0))
        gK = cute.local_tile(mK[batch, kv_head, None, None], (N_BLOCK, D), (None, 0))  # (N_BLOCK, D, n)
        gV = cute.local_tile(mV[batch, kv_head, None, None], (N_BLOCK, D), (None, 0))

        storage = cutlass.memory.SmemAllocator().allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sKV_layout)
        sV = storage.sV.get_tensor(sKV_layout)
        # V transposed, (D, N_BLOCK), as the B operand of P V
        sVt = cute.composition(sV, cute.make_layout((D, N_BLOCK), stride=(N_BLOCK, 1)))

        thr_copy = gmem_copy.get_slice(tidx)
        tQgQ, tQsQ = thr_copy.partition_S(gQ), thr_copy.partition_D(sQ)
        tKgK, tKsK = thr_copy.partition_S(gK), thr_copy.partition_D(sK)
        tVgV, tVsV = thr_copy.partition_S(gV), thr_copy.partition_D(sV)

        thr_mma = tiled_mma.get_slice(tidx)
        tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
        tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
        tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
        acc_O = cute.make_rmem_tensor(thr_mma.partition_shape_C((M_BLOCK, D)), Float32)
        acc_O.fill(0.0)

        # smem to registers with ldmatrix; V is read transposed
        smem_copy_Q = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), elem), tiled_mma)
        smem_copy_K = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), elem), tiled_mma)
        smem_copy_V = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), elem), tiled_mma)
        thr_copy_Q, thr_copy_K = smem_copy_Q.get_slice(tidx), smem_copy_K.get_slice(tidx)
        thr_copy_V = smem_copy_V.get_slice(tidx)

        # Rows past Sq or Sk are zero-filled rather than loaded. The first block of keys is the
        # only one that can run past Sk, since the blocks run from last to first.
        tQcQ = thr_copy.partition_S(cute.local_tile(cute.make_identity_tensor((Sq, D)), (M_BLOCK, D), (m_block, 0)))
        tKVcKV = thr_copy.partition_S(
            cute.local_tile(cute.make_identity_tensor((Sk, D)), (N_BLOCK, D), (n_blocks - 1, 0)))
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            if tQcQ[0, m, 0][0] < Sq:
                cute.copy(gmem_copy, tQgQ[None, m, None], tQsQ[None, m, None])
            else:
                tQsQ[None, m, None].fill(0)
        for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
            if tKVcKV[0, n, 0][0] < Sk:
                cute.copy(gmem_copy, tKgK[None, n, None, n_blocks - 1], tKsK[None, n, None])
            else:
                tKsK[None, n, None].fill(0)
        cute.arch.cp_async_commit_group()

        n_rows = acc_O.shape[0][0] * acc_O.shape[1]  # rows of O this thread holds part of
        row_max = cute.make_rmem_tensor(n_rows, Float32)
        row_sum = cute.make_rmem_tensor(n_rows, Float32)
        row_max.fill(-Float32.inf)
        row_sum.fill(0.0)

        s = SimpleNamespace(
            Sq=Sq, Sk=Sk, m_block=m_block, scale_log2=scale_log2, thr_mma=thr_mma, tiled_mma=tiled_mma,
            gmem_copy=gmem_copy, tKVcKV=tKVcKV, tKgK=tKgK, tKsK=tKsK, tVgV=tVgV, tVsV=tVsV,
            smem_copy_Q=smem_copy_Q, smem_copy_K=smem_copy_K, smem_copy_V=smem_copy_V,
            tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt,
            tSsQ=thr_copy_Q.partition_S(sQ), tSrQ_view=thr_copy_Q.retile(tSrQ),
            tSsK=thr_copy_K.partition_S(sK), tSrK_view=thr_copy_K.retile(tSrK),
            tOsVt=thr_copy_V.partition_S(sVt), tOrVt_view=thr_copy_V.retile(tOrVt),
            acc_O=acc_O, row_max=row_max, row_sum=row_sum,
        )

        # the masked blocks first, then the rest without a mask
        for step in cutlass.range_constexpr(mask_steps):
            n_block = n_blocks - 1 - step
            if n_block >= 0:
                attend_block(s, n_block, is_first=(step == 0), masked=True)
        for step in range(mask_steps, n_blocks, 1):
            attend_block(s, n_blocks - 1 - step, is_first=False, masked=False)

        # O /= the row sum. A row past Sq has a sum of 0; it is not stored.
        acc_O_mn = mn_view(acc_O)
        tOcO_mn = mn_view(thr_mma.partition_C(
            cute.local_tile(cute.make_identity_tensor((Sq, D)), (M_BLOCK, D), (m_block, 0))))
        for r in cutlass.range_constexpr(n_rows):
            total = quad_reduce(row_sum[r], lambda a, b: a + b)
            inv = 1.0 if total == 0.0 else cute.arch.rcp_approx(total)
            acc_O_mn[r, None] = acc_O_mn[r, None].load() * inv
            # log-sum-exp of the scaled scores, from the thread that holds the row's column 0
            row, col = tOcO_mn[r, 0]
            if (col == 0) & (row < Sq):
                mLSE[batch, head, row] = (row_max[r] * scale_log2 + cute.math.log2(total, fastmath=True)) / LOG2_E

        # O through smem (reusing Q's) so that the global stores are 128 bits wide
        rO = cute.make_fragment_like(acc_O, elem)
        rO.store(acc_O.load().to(elem))
        sO = cute.make_tensor(sQ.iterator, sQ_layout)
        smem_copy_O = cute.make_tiled_copy_C(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem), tiled_mma)
        thr_smem_copy_O = smem_copy_O.get_slice(tidx)
        cute.arch.sync_threads()  # every warp is done reading sQ
        cute.copy(smem_copy_O, thr_smem_copy_O.retile(rO), thr_smem_copy_O.partition_D(sO))
        cute.arch.sync_threads()

        gO = cute.local_tile(mO[batch, head, None, None], (M_BLOCK, D), (m_block, 0))
        thr_copy_O = gmem_copy_O.get_slice(tidx)
        tOsO, tOgO = thr_copy_O.partition_S(sO), thr_copy_O.partition_D(gO)
        tOrO = cute.make_fragment_like(tOgO, elem)
        cute.copy(gmem_copy_O, tOsO, tOrO)
        tOcO = thr_copy_O.partition_D(cute.local_tile(cute.make_identity_tensor((Sq, D)), (M_BLOCK, D), (m_block, 0)))
        for m in cutlass.range_constexpr(cute.size(tOgO.shape[1])):
            if tOcO[0, m, 0][0] < Sq:
                cute.copy(gmem_copy_O, tOrO[None, m, None], tOgO[None, m, None])

    @cute.jit
    def launcher(pQ: cute.Pointer, pK: cute.Pointer, pV: cute.Pointer, pO: cute.Pointer, pLSE: cute.Pointer,
                 batch: Int32, heads: Int32, kv_heads: Int32, sq: Int32, sk: Int32, scale_log2: Float32,
                 stream: cuda.CUstream):
        def bhsd(p, h, s):  # torch's contiguous (batch, heads, seq, head_dim)
            return cute.make_tensor(p, cute.make_ordered_layout((batch, h, s, D), order=(3, 2, 1, 0)))

        mQ, mO = bhsd(pQ, heads, sq), bhsd(pO, heads, sq)
        mK, mV = bhsd(pK, kv_heads, sk), bhsd(pV, kv_heads, sk)
        mLSE = cute.make_tensor(pLSE, cute.make_ordered_layout((batch, heads, sq), order=(2, 1, 0)))

        # 8 x 64 smem atoms, swizzled so that ldmatrix reads them without bank conflicts
        atom = cute.make_composed_layout(cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, 64), stride=(64, 1)))
        sQ_layout = cute.tile_to_shape(atom, (M_BLOCK, D), (0, 1))
        sKV_layout = cute.tile_to_shape(atom, (N_BLOCK, D), (0, 1))

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sKV_layout)], 1024]
            sV: cute.struct.Align[cute.struct.MemRange[elem, cute.cosize(sKV_layout)], 1024]

        # global to smem with 128-bit cp.async (and smem to global for O): 8 threads per row of 64
        copy_bits = 128
        per_copy = copy_bits // elem.width
        thr_layout = cute.make_layout((THREADS // 8, 8), stride=(8, 1))
        val_layout = cute.make_layout((1, per_copy))
        gmem_copy = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), elem,
                                num_bits_per_copy=copy_bits), thr_layout, val_layout)
        gmem_copy_O = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem, num_bits_per_copy=copy_bits),
            thr_layout, val_layout)

        tiled_mma = cute.make_tiled_mma(warp.MmaF16BF16Op(elem, Float32, (16, 8, 16)), (THREADS // 32, 1, 1),
                                        permutation_mnk=(THREADS // 32 * 16, 16, 16))

        grid = (cute.ceil_div(sq, M_BLOCK), heads, batch)
        kernel(mQ, mK, mV, mO, mLSE, scale_log2, sQ_layout, sKV_layout, gmem_copy, gmem_copy_O, tiled_mma,
               SharedStorage).launch(grid=grid, block=(THREADS, 1, 1), stream=stream)

    # any address will do: every tensor is promised 16-byte alignment, which the wrapper checks
    p = [make_ptr(e, 4096, cute.AddressSpace.gmem, assumed_align=16) for e in (elem,) * 4 + (Float32,)]
    one = Int32(1)
    return cute.compile(launcher, *p, one, one, one, one, one, Float32(1.0), cuda.CUstream(0))


def attention_fwd_builder(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, is_causal: bool = False,
                          scale: float | None = None) -> Callable:
    """The kernel for these shapes and dtype. q is (batch, heads, Sq, D), k and v are
    (batch, kv_heads, Sk, D), all contiguous, with heads a multiple of kv_heads. Returns
    fn(q, k, v, *, out, lse=None), which writes O to out and, if given, the natural-log
    log-sum-exp of each row's scaled scores to lse, a float32 (batch, heads, Sq) tensor."""
    assert q.dtype in CUTE_DTYPES and q.dtype == k.dtype == v.dtype
    batch, heads, sq, d = q.shape
    _, kv_heads, sk, _ = k.shape
    assert d in (64, 128), "head dims 64 and 128"
    assert k.shape == v.shape == (batch, kv_heads, sk, d) and heads % kv_heads == 0
    assert sq > 0 and sk > 0
    assert q.numel() < 2**31 and k.numel() < 2**31  # the kernel indexes with 32-bit integers
    scale = d**-0.5 if scale is None else scale
    assert scale > 0  # a masked score is -inf before scaling, and must stay -inf after
    shapes = (q.shape, k.shape, v.shape)
    compiled = compile_kernel(q.dtype, d, is_causal)
    elem = CUTE_DTYPES[q.dtype]

    def kernel_wrapper(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, out: torch.Tensor,
                       lse: torch.Tensor | None = None) -> torch.Tensor:
        # The kernel takes pointers and shapes; it does not check them against the tensors.
        assert (q.shape, k.shape, v.shape) == shapes and out.shape == q.shape
        assert q.dtype == k.dtype == v.dtype == out.dtype
        if lse is None:
            lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
        assert lse.shape == q.shape[:3] and lse.dtype == torch.float32
        tensors = (q, k, v, out, lse)
        for t in tensors:
            assert t.is_contiguous() and t.data_ptr() % 16 == 0
        p = [make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
             for t, e in zip(tensors, (elem,) * 4 + (Float32,))]
        compiled(*p, batch, heads, kv_heads, sq, sk, scale * LOG2_E,
                 cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        return out

    return kernel_wrapper
