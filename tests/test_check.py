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
