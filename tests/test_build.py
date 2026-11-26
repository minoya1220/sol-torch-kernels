"""Kernels compile for sm120. Runs without a GPU: CI sets CUTE_DSL_ARCH=sm_120a, so a change
that breaks compiling shows up in CI rather than on the card. `python -m sol_eval.run --all`
compiles every op for every case."""

import pytest
import torch

from sol_eval.cases import AttentionCase, Case
from sol_eval.ops import OPS


def test_add_builds():
    op = OPS["add"]
    inputs = op.make_inputs(Case(63, 1021, torch.bfloat16, layouts=("offset", "row")), torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs))


# each head dim has its own tile sizes, and causal its own mask
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_attention_builds(head_dim, causal):
    op = OPS["attention_fwd"]
    case = AttentionCase(1, 4, 2, 100, 333, head_dim, torch.bfloat16, causal)
    inputs = op.make_inputs(case, torch.Generator().manual_seed(0))
    assert callable(op.build(*inputs, **op.options(case)))


def test_mma_peak_loop_builds():
    from sol_eval.mma_peak import compile_loop
    assert compile_loop() is not None
