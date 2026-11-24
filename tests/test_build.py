"""The kernels compile, for 16-byte aligned rows (the vectorized path) and for inputs that
are not 16-byte aligned. Runs without a GPU: CI sets CUTE_DSL_ARCH=sm_120a so the DSL
targets sm120."""

import pytest
import torch

from sol_eval.cases import Case
from sol_eval.ops import OPS


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", OPS)
def test_builds(name, dtype, offset):
    op = OPS[name]
    inputs = op.make_inputs(Case(63, 1024, dtype, offset), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))
