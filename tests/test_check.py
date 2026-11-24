"""The error rule in sol_eval.check, on small CPU tensors."""

import torch

from sol_eval.check import compare

REF = torch.softmax(torch.randn(8, 100, generator=torch.Generator().manual_seed(0), dtype=torch.float64), dim=-1)


def test_correctly_rounded_result_passes():
    y = REF.to(torch.bfloat16)
    assert compare(y, y.clone(), REF) is None


def test_error_of_many_ulp_fails_when_torch_is_accurate():
    eps = torch.finfo(torch.float32).eps
    y = (REF * (1 + 16 * eps)).float()
    assert compare(y, REF.float(), REF) is not None


def test_nan_where_reference_is_finite_fails():
    y = REF.float()
    y[3, 7] = float("nan")
    assert compare(y, REF.float(), REF) is not None


def test_finite_where_reference_is_nan_fails():
    ref = REF.clone()
    ref[0] = float("nan")  # softmax of a row that is all -inf
    y = ref.float().nan_to_num(0.0)
    assert compare(y, ref.float(), ref) is not None


def test_result_must_overflow_where_the_reference_does():
    # exp(12) is finite in float64 but above float16's largest value, so torch gives inf.
    # Measured as a finite reference, that was an infinite error for torch and so an
    # infinite bound, under which any finite result passed.
    x = torch.tensor([[12.0, 1.0, -1.0, 0.5]], dtype=torch.float16)
    t, ref = torch.exp(x), torch.exp(x.double())
    garbage = torch.where(t.isinf(), t, 3.0).half()
    assert compare(garbage, t, ref) is not None
    assert compare(t.clone(), t, ref) is None


def test_comparison_must_match_exactly():
    a, b = REF[0], REF[1]
    ref = a < b
    y = (a.float() < b.float()).clone()
    assert compare(y, y.clone(), ref) is None
    y[0] = ~y[0]
    assert compare(y, y.clone(), ref) is not None
