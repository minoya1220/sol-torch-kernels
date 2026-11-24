"""The kernels compile, for 16-byte aligned rows (the vectorized path) and for inputs that
are not 16-byte aligned. Runs without a GPU: CI sets CUTE_DSL_ARCH=sm_120a so the DSL
targets sm120."""

import pytest
import torch

from sol_eval.cases import LAYOUTS, Case
from sol_eval.ops import OPS


@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("name", OPS)
def test_builds(name, dtype, offset):
    op = OPS[name]
    inputs = op.make_inputs(Case(63, 1024, dtype, offset), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))


@pytest.mark.parametrize("other", ["contiguous", "same"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_builds_layout(layout, other):
    # with both inputs in the layout, the output takes it too where torch keeps it (transposed)
    layouts = (layout, layout if other == "same" else other)
    op = OPS["add"]
    inputs = op.make_inputs(Case(63, 1021, torch.bfloat16, layouts=layouts), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))
