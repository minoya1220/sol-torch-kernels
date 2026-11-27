"""The ceilings for compute-bound and memory-bound ops. Needs a GPU."""

import math

import pytest
import torch

from sol_eval.measure import ceiling_gbps, measure, peak_flops_per_sm_clock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_mma_rate_is_a_whole_tensor_core_rate():
    # Dense bf16 rates with a float32 accumulator, per SM per clock, are powers of two: 512
    # on GeForce sm_86 and sm_89, and on sm_120 by NVIDIA's ratings; 1024 on the workstation
    # cards of those chips; 2048 on A100. A loop that leaves the tensor cores idle part of the
    # time, or a clock read wrong, lands between them.
    rate, clocks = peak_flops_per_sm_clock()
    assert clocks.sm_mhz > 0
    assert abs(math.log2(rate) - round(math.log2(rate))) < math.log2(1.05), (rate, clocks.sm_mhz)


def test_copy_runs_below_the_memory_ceiling():
    # torch's copy reached 88% of the bus's rate on the RTX 5060 Ti. A wrong bus width, or a
    # memory clock that NVML reports in another unit, puts it above 1 or below one half.
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    us, clocks = measure(lambda x, *, out: out.copy_(x), (src,), torch.empty_like(src))
    share = 2 * src.nbytes / us / 1e3 / ceiling_gbps(clocks.mem_mhz)
    assert 0.5 < share < 1, (share, clocks.mem_mhz)
