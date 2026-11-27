"""The log-sum-exp the attention kernel writes for the backward. The harness checks O; this
checks the second output. Needs a GPU."""

import pytest
import torch

from sol_eval.cases import AttentionCase, make_attention_inputs
from sol_kernels.attention_fwd import attention_fwd_builder

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("causal", (False, True))
def test_lse_matches_float64(causal):
    case = AttentionCase(2, 8, 2, 257, 1000, 128, torch.bfloat16, causal, scale=0.2)
    q, k, v = make_attention_inputs(case, torch.Generator("cuda").manual_seed(0))
    lse = torch.empty(q.shape[:3], dtype=torch.float32, device="cuda")
    attention_fwd_builder(q, k, v, is_causal=causal, scale=0.2)(q, k, v, out=torch.empty_like(q), lse=lse)

    s = q.double() @ k.double().repeat_interleave(4, dim=1).transpose(-1, -2) * 0.2
    if causal:
        s.masked_fill_(torch.ones(257, 1000, dtype=torch.bool, device="cuda").tril().logical_not(), float("-inf"))
    # The backward recomputes P = exp(S - lse), so an error d in lse scales P by exp(d).
    # 2^-12 keeps that under 1/32 of a bf16 ulp; float32 scores and sums are far closer.
    assert (lse.double() - s.logsumexp(dim=-1)).abs().max().item() < 2**-12
