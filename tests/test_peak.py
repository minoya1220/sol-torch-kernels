"""The ceiling for compute-bound ops. Needs a GPU."""

import math

import pytest
import torch

from sol_eval.measure import peak_flops_per_sm_clock

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def test_mma_rate_is_a_whole_tensor_core_rate():
    # Dense bf16 rates with a float32 accumulator, per SM per clock, are powers of two: 512
    # on GeForce sm_86 and sm_89, and on sm_120 by NVIDIA's ratings; 1024 on the workstation
    # cards of those chips; 2048 on A100. A loop that leaves the tensor cores idle part of the
    # time, or a clock read wrong, lands between them.
    rate, clocks = peak_flops_per_sm_clock()
    assert clocks.sm_mhz > 0
    assert abs(math.log2(rate) - round(math.log2(rate))) < math.log2(1.05), (rate, clocks.sm_mhz)
