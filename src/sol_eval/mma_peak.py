"""Tensor-core FLOPs per SM per clock, measured with a loop of the m16n8k16 bf16 mma.sync
that the attention kernel uses, accumulating in float32. A kernel cannot issue that
instruction faster than this loop, so the loop's rate times the SM clock is the ceiling for
compute-bound ops. GeForce cards run the float32 accumulator at half the rate of the
workstation cards of the same chip family; measuring settles which rate this card has."""

from collections.abc import Callable
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_ptr

M, N, K = 64, 64, 16  # one warp's tile per gemm call: 4 x 8 independent mma.sync, 4096 FLOP each
WARPS_PER_SM = 8  # 167 registers per thread: 8 warps fit on an SM at once, 16 would not
ITERS = 8192  # gemm calls per warp per launch: about 6 ms on an RTX 5060 Ti, so one timing
# spans several of NVML's clock sample periods


@lru_cache(maxsize=None)
def compile_loop():
    @cute.kernel
    def kernel(mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, iters: Int32, tiled_mma: cute.TiledMma):
        tidx, _, _ = cute.arch.thread_idx()
        thr = tiled_mma.get_slice(tidx)
        rA = thr.make_fragment_A(thr.partition_A(mA))
        rB = thr.make_fragment_B(thr.partition_B(mB))
        cute.autovec_copy(thr.partition_A(mA), rA)
        cute.autovec_copy(thr.partition_B(mB), rB)
        acc = cute.make_rmem_tensor(thr.partition_shape_C((M, N)), Float32)
        acc.fill(0.0)
        for _ in range(0, iters, 1):
            cute.gemm(tiled_mma, acc, rA, rB, acc)
        # every warp stores to the same place: the store only keeps the loop from being removed
        cute.autovec_copy(acc, thr.partition_C(mC))

    @cute.jit
    def launcher(pA: cute.Pointer, pB: cute.Pointer, pC: cute.Pointer, blocks: Int32, iters: Int32,
                 stream: cuda.CUstream):
        mA = cute.make_tensor(pA, cute.make_layout((M, K), stride=(K, 1)))
        mB = cute.make_tensor(pB, cute.make_layout((N, K), stride=(K, 1)))
        mC = cute.make_tensor(pC, cute.make_layout((M, N), stride=(N, 1)))
        tiled_mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, Float32, (16, 8, 16)), (1, 1, 1))
        kernel(mA, mB, mC, iters, tiled_mma).launch(grid=(blocks, 1, 1), block=(32, 1, 1), stream=stream)

    p = [make_ptr(e, 4096, cute.AddressSpace.gmem, assumed_align=16)
         for e in (cutlass.BFloat16, cutlass.BFloat16, Float32)]
    return cute.compile(launcher, *p, Int32(1), Int32(1), cuda.CUstream(0))


def mma_loop() -> tuple[tuple, torch.Tensor, int, Callable]:
    """Inputs, output, FLOPs per call, and fn(a, b, *, out) that runs the loop once on every
    SM, in the form the harness times."""
    compiled = compile_loop()
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    blocks = sms * WARPS_PER_SM
    a, b = (torch.randn(r, K, device="cuda").bfloat16() for r in (M, N))
    out = torch.empty(M, N, device="cuda")

    def fn(a, b, *, out):
        ptrs = [make_ptr(e, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                for e, t in ((cutlass.BFloat16, a), (cutlass.BFloat16, b), (Float32, out))]
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled(*ptrs, Int32(blocks), Int32(ITERS), stream)

    return (a, b), out, blocks * ITERS * 2 * M * N * K, fn
