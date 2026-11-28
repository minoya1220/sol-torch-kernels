"""The error rule in sol_eval.check, on small CPU tensors."""

import torch

from sol_eval.cases import Case
from sol_eval.check import compare
from sol_eval.ops import OPS

REF = torch.softmax(torch.randn(8, 100, generator=torch.Generator().manual_seed(0), dtype=torch.float64), dim=-1)


def test_correctly_rounded_result_passes():
    y = REF.to(torch.bfloat16)
    assert compare(y, y.clone(), REF, "max_scaled") is None


def test_error_of_many_ulp_fails_when_torch_is_accurate():
    eps = torch.finfo(torch.float32).eps
    y = (REF * (1 + 16 * eps)).float()
    assert compare(y, REF.float(), REF, "max_scaled") is not None


def test_nan_where_reference_is_finite_fails():
    y = REF.float()
    y[3, 7] = float("nan")
    assert compare(y, REF.float(), REF, "max_scaled") is not None


def test_finite_where_reference_is_nan_fails():
    ref = REF.clone()
    ref[0] = float("nan")  # softmax of a row that is all -inf
    y = ref.float().nan_to_num(0.0)
    assert compare(y, ref.float(), ref, "max_scaled") is not None


def test_result_must_overflow_where_the_reference_does():
    # exp(12) is finite in float64 but above float16's largest value, so torch gives inf.
    # Measured as a finite reference, that was an infinite error for torch and so an
    # infinite bound, under which any finite result passed.
    x = torch.tensor([[12.0, 1.0, -1.0, 0.5]], dtype=torch.float16)
    t, ref = torch.exp(x), torch.exp(x.double())
    garbage = torch.where(t.isinf(), t, 3.0).half()
    assert compare(garbage, t, ref, "relative") is not None
    assert compare(t.clone(), t, ref, "relative") is None


def test_comparison_must_match_torch_exactly():
    a, b = REF[0], REF[1]
    t = a.float() < b.float()
    y = t.clone()
    assert compare(y, t, a < b, "exact") is None
    y[0] = ~y[0]
    assert compare(y, t, a < b, "exact") is not None


def test_one_ulp_off_fails_for_an_exact_op():
    # a + b is one correctly rounded step, so there is exactly one right answer
    gen = torch.Generator().manual_seed(0)
    a, b = (torch.randn(1000, generator=gen).bfloat16() for _ in range(2))
    t = a + b
    y = t.clone()
    y.view(torch.int16)[0] += 1
    assert compare(y, t, a.double() + b.double(), "exact") is not None


def test_small_results_are_measured_against_themselves():
    # Zeroing every product below 1e-3 passes when small entries are measured against the
    # largest one, as for softmax. An elementwise output stands alone, so it must fail.
    gen = torch.Generator().manual_seed(0)
    a, b = (torch.randn(63, 1021, generator=gen).bfloat16() for _ in range(2))
    t, ref = a * b, a.double() * b.double()
    y = torch.where(t.abs() < 1e-3, 0.0, t).bfloat16()
    assert compare(y, t, ref, "max_scaled") is None
    assert compare(y, t, ref, "relative") is not None


def test_one_hard_element_does_not_loosen_the_bound_for_the_others():
    # Where x is very negative, torch's float32 gelu and silu lose every digit (silu(-100)
    # gives -0, not about -4e-42), so torch's error there is about 1/eps. Compared as
    # maxima over the tensor, that let a result 2^-10 off on every element pass.
    x = torch.randn(63, 1021, generator=torch.Generator().manual_seed(0)) * 100
    for fn in (torch.nn.functional.gelu, torch.nn.functional.silu):
        t, ref = fn(x), fn(x.double())
        assert compare(t.clone(), t, ref, "relative") is None
        assert compare(t * (1 + 2**-10), t, ref, "relative") is not None


def sequential_sum(x, dtype):
    s = torch.zeros(x.shape[0], dtype=dtype)
    for k in range(x.shape[1]):
        s = (s.float() + x[:, k]).to(dtype)
    return s.float()


def backward_with_sum(Y, dY, dtype):
    """Softmax backward with the row sum taken left to right and kept in dtype."""
    S = sequential_sum(Y.float() * dY.float(), dtype)
    return (Y.float() * (dY.float() - S[:, None])).to(Y.dtype)


def backward_case(dtype):
    op = OPS["softmax_bwd"]
    Y, dY = op.make_inputs(Case(63, 1021, dtype), torch.Generator().manual_seed(0))
    t = op.make_out(Y, dY)
    op.torch_fn(Y, dY, out=t)
    return Y, dY, t, op.ref64(Y, dY), op.sum_size(Y, dY)


def test_a_float32_sum_in_another_order_passes_the_backward():
    # Where dY_i is close to the row sum, dX_i carries the sum's error. Against torch's error on
    # that element alone, a float32 sum in another order than torch's fails wherever torch's
    # sum happened to land close; against the size of the summed terms, it passes.
    Y, dY, t, ref, sum_size = backward_case(torch.float32)
    y = backward_with_sum(Y, dY, torch.float32)
    assert compare(y, t, ref, "max_scaled") is not None
    assert compare(y, t, ref, "sum_scaled", sum_size) is None


def test_a_16_bit_sum_fails_the_backward():
    Y, dY, t, ref, sum_size = backward_case(torch.bfloat16)
    assert compare(backward_with_sum(Y, dY, torch.float32), t, ref, "sum_scaled", sum_size) is None
    assert compare(backward_with_sum(Y, dY, torch.bfloat16), t, ref, "sum_scaled", sum_size) is not None


def sum_case(dtype):
    """Rows of normal values whose last term cancels the rest, so each row's sum is close to 0
    and its float32 rounding error is large against the sum itself."""
    op = OPS["sum"]
    x = torch.randn(63, 4099, generator=torch.Generator().manual_seed(0)).to(dtype)
    x[:, -1] = 0
    x[:, -1] = (-x.double().sum(-1)).to(dtype)
    o = {"dim": -1, "keepdim": False}
    (t,) = op.make_out(x, **o)
    op.torch_fn(x, out=(t,), **o)
    return x, t, op.ref64(x, **o)[0], op.sum_size(x, **o)[0]


def blocked_sum(x, dtype, chains=256):
    """A row sum as a kernel takes it: chains partial sums, each over every chains-th element
    and kept in dtype, then added left to right."""
    s = torch.zeros(x.shape[0], chains, dtype=dtype)
    x = torch.nn.functional.pad(x, (0, -x.shape[1] % chains)).view(x.shape[0], -1, chains)
    for k in range(x.shape[1]):
        s = (s.float() + x[:, k].float()).to(dtype)
    return sequential_sum(s, dtype).to(x.dtype)


def test_a_float32_sum_in_another_order_passes_a_reduction():
    # Against torch's error on each row alone, a float32 sum in another order than torch's
    # fails wherever torch's sum happened to land close; against the size of the terms, it passes.
    x, t, ref, sum_size = sum_case(torch.float32)
    y = blocked_sum(x, torch.float32)
    assert compare(y, t, ref, "relative") is not None
    assert compare(y, t, ref, "sum_scaled", sum_size) is None


def test_a_16_bit_sum_fails_a_reduction():
    x, t, ref, sum_size = sum_case(torch.bfloat16)
    assert compare(blocked_sum(x, torch.float32), t, ref, "sum_scaled", sum_size) is None
    assert compare(blocked_sum(x, torch.bfloat16), t, ref, "sum_scaled", sum_size) is not None


def test_a_variance_from_the_mean_of_squares_fails():
    # E[x^2] - E[x]^2 loses its digits when the mean is far from 0; Welford's update does not
    op = OPS["var"]
    x = torch.randn(63, 1021, generator=torch.Generator().manual_seed(0)) + 1000
    o = {"dim": -1, "keepdim": False, "correction": 1}
    (t,) = op.make_out(x, **o)
    op.torch_fn(x, out=(t,), **o)
    ref, sum_size = op.ref64(x, **o)[0], op.sum_size(x, **o)[0]
    n = x.shape[1]
    two_pass = (x - x.mean(-1, keepdim=True)).square().sum(-1) / (n - 1)
    naive = (x.square().sum(-1) - x.sum(-1).square() / n) / (n - 1)
    assert compare(two_pass, t, ref, "sum_scaled", sum_size) is None
    assert compare(naive, t, ref, "sum_scaled", sum_size) is not None
