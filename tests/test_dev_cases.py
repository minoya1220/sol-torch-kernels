"""Every dev case passes, for both softmax ops and three elementwise ops. The rest of
sol_kernels.elementwise_ops is checked with `python -m sol_eval.run <op>`, because
compiling every op for every case takes too long for a test run. Needs a GPU."""

import pytest
import torch

from sol_eval.check import check
from sol_eval.ops import OPS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NAMES = ["softmax_fwd", "softmax_bwd", "add", "mul", "exp"]


@pytest.mark.parametrize("name,case", [(n, c) for n in NAMES for c in OPS[n].dev_cases()],
                         ids=lambda x: str(x))
def test_dev_case(name, case):
    op = OPS[name]
    inputs = op.make_inputs(case, torch.Generator("cuda").manual_seed(case.seed))
    assert check(op, op.build(*inputs), inputs) is None
