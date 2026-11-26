"""The error rule in sol_eval.check, on small CPU tensors."""

import torch

from sol_eval.cases import AttentionCase, Case
from sol_eval.check import compare
from sol_eval.ops import OPS, attention_probs64

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


def flash_attention(q, k, v, *, drop_keys=0, acc_dtype=torch.float32):
    """Flash-attention forward as a correct 16-bit kernel computes it: blocks of 64 keys, an
    online softmax in float32, P rounded to the input dtype before P V, and O accumulated in
    acc_dtype. drop_keys leaves out that many keys at the end."""
    d, sk = q.shape[-1], k.shape[2]
    s_all = q.float() @ k.float().transpose(-1, -2) * d**-0.5
    m = torch.full(q.shape[:3] + (1,), float("-inf"))
    l = torch.zeros(q.shape[:3] + (1,))
    o = torch.zeros(q.shape, dtype=acc_dtype)
    for j in range(0, sk - drop_keys, 64):
        s = s_all[..., j:min(j + 64, sk - drop_keys)]
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        p, correction = torch.exp(s - m_new), torch.exp(m - m_new)
        l = l * correction + p.sum(-1, keepdim=True)
        o = (o.float() * correction + p.to(q.dtype).float() @ v[..., j:j + p.shape[-1], :].float()).to(acc_dtype)
        m = m_new
    return (o.float() / l).to(q.dtype)


def attention_case():
    """A long row of keys, where rounding error grows with their number."""
    op = OPS["attention_fwd"]
    case = AttentionCase(1, 2, 2, 64, 4099, 64, torch.bfloat16)
    q, k, v = op.make_inputs(case, torch.Generator().manual_seed(0))
    options = op.options(case)
    t = op.make_out(q, k, v)
    op.torch_fn(q, k, v, out=t, **options)
    return q, k, v, t, op.ref64(q, k, v, **options), op.sum_size(q, k, v, **options), options


def test_a_correct_flash_kernel_passes_attention():
    # Measured against |r| alone, it fails where O is near 0: it rounds P at other points than
    # torch does, and the two errors there are unrelated.
    q, k, v, t, ref, rss, _ = attention_case()
    y = flash_attention(q, k, v)
    assert compare(y, t, ref, "relative") is not None
    assert compare(y, t, ref, "rss_scaled", rss) is None


def test_dropping_one_key_fails_attention():
    # Against the sum of the terms' sizes, as for the softmax backward, a kernel that drops
    # 8 of 4099 keys passes: P's roundings add up like a random walk, to the root sum of
    # squares, which is about sqrt(Sk) times smaller.
    q, k, v, t, ref, rss, options = attention_case()
    sizes = torch.empty(q.shape, dtype=torch.float64)
    for b, h, rows, p, vh in attention_probs64(q, k, v, **options):
        sizes[b, h, rows] = p @ vh.abs()
    assert compare(flash_attention(q, k, v, drop_keys=8), t, ref, "rss_scaled", sizes) is None
    assert compare(flash_attention(q, k, v, drop_keys=1), t, ref, "rss_scaled", rss) is not None


def test_a_16_bit_accumulator_fails_attention():
    q, k, v, t, ref, rss, _ = attention_case()
    assert compare(flash_attention(q, k, v, acc_dtype=torch.bfloat16), t, ref, "rss_scaled", rss) is not None
