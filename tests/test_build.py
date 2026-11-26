"""add compiles for sm120, for every dtype the builder stores. Runs without a GPU: CI sets CUTE_DSL_ARCH=sm_120a, so a change
that breaks compiling shows up in CI rather than on the card. `python -m sol_eval.run --all`
compiles every op for every case."""

import pytest
import torch

from sol_eval.cases import EXTRA_DTYPES, Case
from sol_eval.ops import OPS


def test_add_builds():
    op = OPS["add"]
    inputs = op.make_inputs(Case(63, 1021, torch.bfloat16, layouts=("offset", "row")), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))


@pytest.mark.parametrize("dtype", EXTRA_DTYPES)
def test_add_builds_for_other_dtypes(dtype):
    # float64, the integer types and bool load, compute and store in their own types
    op = OPS["add"]
    inputs = op.make_inputs(Case(63, 1021, dtype, layouts=("offset", "row")), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))
