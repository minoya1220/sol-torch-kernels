"""add compiles for sm120. Runs without a GPU: CI sets CUTE_DSL_ARCH=sm_120a, so a change
that breaks compiling shows up in CI rather than on the card. `python -m sol_eval.run --all`
compiles every op for every case."""

import torch

from sol_eval.cases import Case, ReductionCase
from sol_eval.ops import OPS


def test_add_builds():
    op = OPS["add"]
    inputs = op.make_inputs(Case(63, 1021, torch.bfloat16, layouts=("offset", "row")), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))


def test_reductions_build():
    # one case per kind of variant: rows, columns, the split path, argmax's index, a fused norm
    for name, case in [("sum", ReductionCase((63, 1021), -1, torch.bfloat16, layout="offset")),
                       ("sum", ReductionCase((1021, 63), 0, torch.float16)),
                       ("var", ReductionCase((2, 2**20 + 3), -1, torch.float32)),
                       ("argmax", ReductionCase((7, 1021, 9), 1, torch.bfloat16, layout="strided")),
                       ("all", ReductionCase((63, 1021), None, torch.bool))]:
        op = OPS[name]
        inputs = op.make_inputs(case, torch.Generator().manual_seed(0))
        assert callable(op.build(*inputs, **op.options(case)))
    for name in ("softmax_fwd", "softmax_bwd", "layer_norm", "rms_norm"):
        op = OPS[name]
        assert callable(op.build(*op.make_inputs(Case(63, 4099, torch.bfloat16, offset=1), torch.Generator().manual_seed(0))))
