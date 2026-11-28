"""Torch reductions written as states and combine functions for reduction_builder.

Each entry gives the parts of its state, the identity state, how one element is folded in,
how two states merge, how a state turns into the values stored, and torch's call, which the
kernel must match. Every function takes and returns DSL values: Float32 for values and counts,
Int32 for indices. sol_eval.ops turns each entry into an op the harness checks and times.

The fused ops (softmax and its backward, layer_norm, rms_norm) also give an epilogue, which
computes one element of the full-size output from the inputs there and the finished values of
its row. Their entries are at the end.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from sol_kernels.reduction import reduction_builder

aten = torch.ops.aten
INF = float("inf")


@dataclass(frozen=True)
class ReductionOp:
    name: str
    state: tuple         # the DSL type of each part of the state
    init: Callable       # () -> the state that combine leaves unchanged
    update: Callable     # (state, xs, i) -> the state with one element folded in. xs holds the
                         # element of each reduction input, i its index along the reduced dims
    combine: Callable    # (state, state) -> the merged state
    finish: Callable     # (state, n, params) -> the values stored, for n elements reduced
    torch_call: Callable  # (*inputs, **options) -> torch's result
    params: Callable = lambda **options: ()  # the options the kernel takes, as floats
    options: tuple = ()  # the names of the options beyond dim and keepdim
    reduce_inputs: int = 1  # how many of the inputs the reduction reads
    epilogue: Callable | None = None  # (xs, values, params) -> one element of the full-size output
    bool_input: bool = False  # also checked on bool inputs


def select(cond, a, b):
    return Float32(cutlass.select_(cond, Float32(a), Float32(b)))


# torch's max and min of a NaN are NaN
def fmax(a, b):
    return cute.math.max(Float32(a), Float32(b), propagate_nan=True)


def fmin(a, b):
    return cute.math.min(Float32(a), Float32(b), propagate_nan=True)


def one_part(f):
    """update and combine for a state of one Float32 merged with f."""
    return (lambda s, xs, i: (f(s[0], xs[0]),)), (lambda a, b: (f(a[0], b[0]),))


def first(s, n, params):
    return (s[0],)


def add(a, b):
    return a + b


# max and min with their index, as torch picks them: the first NaN, else the first of the
# largest (or smallest) values. The state is (value, index).
def picks_max(a, b):
    """Whether b's element wins over a's in torch.max."""
    a_ok, b_ok = a[0] == a[0], b[0] == b[0]
    lower = b[1] < a[1]
    return ((b[0] != b[0]) & (a_ok | lower)) | (a_ok & b_ok & ((b[0] > a[0]) | ((b[0] == a[0]) & lower)))


def picks_min(a, b):
    a_ok, b_ok = a[0] == a[0], b[0] == b[0]
    lower = b[1] < a[1]
    return ((b[0] != b[0]) & (a_ok | lower)) | (a_ok & b_ok & ((b[0] < a[0]) | ((b[0] == a[0]) & lower)))


def pick(picks):
    def combine(a, b):
        p = picks(a, b)
        return select(p, b[0], a[0]), Int32(cutlass.select_(p, Int32(b[1]), Int32(a[1])))

    return combine


combine_max, combine_min = pick(picks_max), pick(picks_min)
MAX_INDEX = 2**31 - 1  # the identity's index loses every tie


# Welford's running count, mean and sum of squared deviations, merged as in Chan et al.
def welford_update(s, xs, i):
    n = s[0] + 1.0
    d = xs[0] - s[1]
    mean = s[1] + d / n
    return n, mean, s[2] + d * (xs[0] - mean)


def welford_combine(a, b):
    n = a[0] + b[0]
    d = b[1] - a[1]
    w = select(n == 0.0, 0.0, b[0] / n)  # b's share of the count
    return n, a[1] + d * w, a[2] + b[2] + d * d * a[0] * w


def variance(s, params):  # torch divides by max(n - correction, 0)
    return s[2] / fmax(s[0] - params[0], 0.0)


# var_mean and std_mean also keep a plain sum for the mean, as torch does: Welford's mean
# turns NaN at the first inf, where torch's mean is inf
def welford_sum_update(s, xs, i):
    return *welford_update(s, xs, i), s[3] + xs[0]


def welford_sum_combine(a, b):
    return *welford_combine(a, b), a[3] + b[3]


# log-sum-exp as a running max m and sum of exp(x - m)
def lse_update(s, xs, i):
    x = xs[0]
    m = fmax(s[0], x)
    small = fmin(s[0], x)
    e = select(small == m, 1.0, cute.math.exp(small - m))  # also when both are -inf or inf
    return m, select(x > s[0], s[1] * e + 1.0, s[1] + e)


def lse_combine(a, b):
    m = fmax(a[0], b[0])
    sa = select(a[0] == m, a[1], a[1] * cute.math.exp(a[0] - m))
    sb = select(b[0] == m, b[1], b[1] * cute.math.exp(b[0] - m))
    return m, sa + sb


def dims_or_all(dim):
    return () if dim is None else dim


WELFORD = (Float32, Float32, Float32)
CORRECTION = {"params": lambda correction, **_: (float(correction),), "options": ("correction",)}

REDUCTION_OPS = [
    ReductionOp("sum", (Float32,), lambda: (Float32(0.0),), *one_part(add), first,
                lambda x, dim, keepdim: torch.sum(x, dim, keepdim=keepdim)),
    ReductionOp("mean", (Float32,), lambda: (Float32(0.0),), *one_part(add), lambda s, n, p: (s[0] * (1.0 / n),),
                lambda x, dim, keepdim: torch.mean(x, dim, keepdim=keepdim)),
    ReductionOp("amax", (Float32,), lambda: (Float32(-INF),), *one_part(fmax), first,
                lambda x, dim, keepdim: torch.amax(x, dims_or_all(dim), keepdim=keepdim)),
    ReductionOp("amin", (Float32,), lambda: (Float32(INF),), *one_part(fmin), first,
                lambda x, dim, keepdim: torch.amin(x, dims_or_all(dim), keepdim=keepdim)),
    ReductionOp("max", (Float32, Int32), lambda: (Float32(-INF), Int32(MAX_INDEX)),
                lambda s, xs, i: combine_max(s, (xs[0], i)), combine_max, lambda s, n, p: s,
                lambda x, dim, keepdim: tuple(torch.max(x, dim, keepdim=keepdim))),
    ReductionOp("min", (Float32, Int32), lambda: (Float32(INF), Int32(MAX_INDEX)),
                lambda s, xs, i: combine_min(s, (xs[0], i)), combine_min, lambda s, n, p: s,
                lambda x, dim, keepdim: tuple(torch.min(x, dim, keepdim=keepdim))),
    ReductionOp("argmax", (Float32, Int32), lambda: (Float32(-INF), Int32(MAX_INDEX)),
                lambda s, xs, i: combine_max(s, (xs[0], i)), combine_max, lambda s, n, p: (s[1],),
                lambda x, dim, keepdim: torch.argmax(x, dim, keepdim=keepdim)),
    ReductionOp("argmin", (Float32, Int32), lambda: (Float32(INF), Int32(MAX_INDEX)),
                lambda s, xs, i: combine_min(s, (xs[0], i)), combine_min, lambda s, n, p: (s[1],),
                lambda x, dim, keepdim: torch.argmin(x, dim, keepdim=keepdim)),
    ReductionOp("var", WELFORD, lambda: (Float32(0.0),) * 3, welford_update, welford_combine,
                lambda s, n, p: (variance(s, p),),
                lambda x, dim, keepdim, correction: torch.var(x, dim, correction=correction, keepdim=keepdim),
                **CORRECTION),
    ReductionOp("std", WELFORD, lambda: (Float32(0.0),) * 3, welford_update, welford_combine,
                lambda s, n, p: (cute.math.sqrt(variance(s, p)),),
                lambda x, dim, keepdim, correction: torch.std(x, dim, correction=correction, keepdim=keepdim),
                **CORRECTION),
    ReductionOp("var_mean", WELFORD + (Float32,), lambda: (Float32(0.0),) * 4, welford_sum_update, welford_sum_combine,
                lambda s, n, p: (variance(s, p), s[3] * (1.0 / n)),
                lambda x, dim, keepdim, correction: torch.var_mean(x, dim, correction=correction, keepdim=keepdim),
                **CORRECTION),
    ReductionOp("std_mean", WELFORD + (Float32,), lambda: (Float32(0.0),) * 4, welford_sum_update, welford_sum_combine,
                lambda s, n, p: (cute.math.sqrt(variance(s, p)), s[3] * (1.0 / n)),
                lambda x, dim, keepdim, correction: torch.std_mean(x, dim, correction=correction, keepdim=keepdim),
                **CORRECTION),
    ReductionOp("norm1", (Float32,), lambda: (Float32(0.0),), *one_part(lambda s, x: s + cute.math.abs(x)),
                lambda s, n, p: (s[0],), lambda x, dim, keepdim: torch.linalg.vector_norm(x, 1, dim, keepdim)),
    ReductionOp("norm2", (Float32,), lambda: (Float32(0.0),), lambda s, xs, i: (s[0] + xs[0] * xs[0],),
                lambda a, b: (a[0] + b[0],), lambda s, n, p: (cute.math.sqrt(s[0]),),
                lambda x, dim, keepdim: torch.linalg.vector_norm(x, 2, dim, keepdim)),
    ReductionOp("norm_inf", (Float32,), lambda: (Float32(0.0),), lambda s, xs, i: (fmax(s[0], cute.math.abs(xs[0])),),
                lambda a, b: (fmax(a[0], b[0]),), first,
                lambda x, dim, keepdim: torch.linalg.vector_norm(x, INF, dim, keepdim)),
    ReductionOp("logsumexp", (Float32, Float32), lambda: (Float32(-INF), Float32(0.0)), lse_update, lse_combine,
                lambda s, n, p: (s[0] + cute.math.log(s[1]),),
                lambda x, dim, keepdim: torch.logsumexp(x, dims_or_all(dim), keepdim=keepdim)),
    # all and any keep 1.0 for true and 0.0 for false, which the bool output stores as 1 and 0
    ReductionOp("all", (Float32,), lambda: (Float32(1.0),), lambda s, xs, i: (select(xs[0] != 0.0, s[0], 0.0),),
                lambda a, b: (cute.math.min(a[0], b[0]),), first,
                lambda x, dim, keepdim: torch.all(x, dim, keepdim=keepdim), bool_input=True),
    ReductionOp("any", (Float32,), lambda: (Float32(0.0),), lambda s, xs, i: (select(xs[0] != 0.0, 1.0, s[0]),),
                lambda a, b: (cute.math.max(a[0], b[0]),), first,
                lambda x, dim, keepdim: torch.any(x, dim, keepdim=keepdim), bool_input=True),
]


def softmax_finish(s, n, params):
    return s[0], 1.0 / s[1]


def layer_norm_finish(s, n, params):
    return s[1], cute.math.rsqrt(s[2] / n + params[0])


# The fused ops. Each one's torch call is in sol_eval.ops, since its inputs and options differ.
SOFTMAX_FWD = ReductionOp("softmax_fwd", (Float32, Float32), lambda: (Float32(-INF), Float32(0.0)), lse_update,
                          lse_combine, softmax_finish, None,
                          epilogue=lambda xs, v, p: cute.math.exp(xs[0] - v[0]) * v[1])
SOFTMAX_BWD = ReductionOp("softmax_bwd", (Float32,), lambda: (Float32(0.0),), lambda s, xs, i: (s[0] + xs[0] * xs[1],),
                          lambda a, b: (a[0] + b[0],), first, None, reduce_inputs=2,
                          epilogue=lambda xs, v, p: xs[0] * (xs[1] - v[0]))
LAYER_NORM = ReductionOp("layer_norm", WELFORD, lambda: (Float32(0.0),) * 3, welford_update, welford_combine,
                         layer_norm_finish, None,
                         epilogue=lambda xs, v, p: (xs[0] - v[0]) * v[1] * xs[1] + xs[2])
RMS_NORM = ReductionOp("rms_norm", (Float32,), lambda: (Float32(0.0),), lambda s, xs, i: (s[0] + xs[0] * xs[0],),
                       lambda a, b: (a[0] + b[0],), lambda s, n, p: (cute.math.rsqrt(s[0] / n + p[0]),), None,
                       epilogue=lambda xs, v, p: xs[0] * v[0] * xs[1])


def softmax_fwd_builder(X: torch.Tensor):
    """Softmax over the last dim of X, the call the harness and the exploits use."""
    return reduction_builder(SOFTMAX_FWD, X, dims=(X.dim() - 1,), outs_like=(torch.empty_like(X, device="meta"),))


def softmax_bwd_builder(Y: torch.Tensor, dY: torch.Tensor):
    """The gradient of softmax over the last dim, from its output Y and the output's gradient dY."""
    return reduction_builder(SOFTMAX_BWD, Y, dY, dims=(Y.dim() - 1,), outs_like=(torch.empty_like(Y, device="meta"),))
