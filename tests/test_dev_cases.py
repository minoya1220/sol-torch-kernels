"""Every dev case passes, for both ops. Needs a GPU."""

import pytest
import torch

from sol_eval.cases import dev_cases
from sol_eval.check import check
from sol_eval.ops import OPS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("case", dev_cases(), ids=str)
@pytest.mark.parametrize("name", OPS)
def test_dev_case(name, case):
    op = OPS[name]
    inputs = op.make_inputs(case, torch.Generator("cuda").manual_seed(case.seed))
    assert check(op, op.build(*inputs), inputs) is None
