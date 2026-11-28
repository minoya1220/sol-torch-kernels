"""The ops the harness can check and time. Each says how to make inputs for a case,
how torch computes it in the kernel's call form and in its usual call, its float64
reference, the bytes it must move, how to allocate its output, which error rule it is
checked with, and its cases. An op with several outputs (max with its indices, layer_norm
with its saved statistics) has a tuple of outputs, references and rules, one per output."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from math import prod

import torch

from sol_eval.cases import (Case, ReductionCase, dev_cases, elementwise_dev_cases, elementwise_heldout_cases,
                            heldout_cases, make_input, make_reduction_input, make_tensor, reduction_dev_cases,
                            reduction_heldout_cases)
from sol_kernels.elementwise import elementwise_builder
from sol_kernels.elementwise_ops import ELEMENTWISE_OPS, ElementwiseOp
from sol_kernels.reduction import reduction_builder
from sol_kernels.reduction_ops import (LAYER_NORM, REDUCTION_OPS, RMS_NORM, ReductionOp, softmax_bwd_builder,
                                       softmax_fwd_builder)


@dataclass(frozen=True)
class Op:
    name: str
    build: Callable        # build(*inputs) -> fn, and fn(*inputs, out=) writes out
    make_inputs: Callable  # (case, generator) -> tuple of input tensors
    torch_fn: Callable     # torch in the same call form as fn, for the checks
    torch_call: Callable   # torch's usual call, without out=, which is what is timed
    ref64: Callable        # (*inputs) -> float64 result
    nbytes: Callable       # (*inputs, **options) -> bytes the op must read and write
    make_out: Callable     # (*inputs, **options) -> an empty output for fn to write
    rule: str | tuple      # the error rule in sol_eval.check, or one per output
    dev_cases: Callable    # () -> correctness cases
    heldout_cases: Callable
    perf_cases: list       # cases to time
    sum_size: Callable | None = None  # (*inputs) -> per element, the size of the terms of the sum it uses
    # Keyword arguments of the call that are not tensors, such as a reduction's dim and
    # keepdim. They go to build, torch_fn, torch_call, ref64, sum_size, make_out and nbytes.
    options: Callable = lambda case: {}


# 2^27 bf16 elements is 256 MiB per tensor, far above L2, while the float64
# reference for the softmax backward still fits on a 16 GB card
PERF_ELEMENTS = 2**27
SOFTMAX_PERF_CASES = [Case(PERF_ELEMENTS // n, n, torch.bfloat16)
                      for n in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)]


def fwd_inputs(case: Case, gen: torch.Generator):
    return (make_tensor(case, gen),)


def fwd_torch(X, *, out):
    return torch.ops.aten._softmax.out(X, -1, False, out=out)


def fwd_torch_call(X):
    return torch.softmax(X, -1)


def fwd_ref64(X):
    return torch.softmax(X.double(), dim=-1)


def bwd_inputs(case: Case, gen: torch.Generator):
    # Y must be a real softmax output; it is written in place to keep the case's offset
    Y = make_tensor(case, gen)
    Y.copy_(torch.softmax(Y, dim=-1))
    dY = make_tensor(replace(case, values="normal"), gen)
    return Y, dY


def bwd_torch(Y, dY, *, out):
    return torch.ops.aten._softmax_backward_data.out(dY, Y, -1, Y.dtype, grad_input=out)


def bwd_torch_call(Y, dY):
    return torch.ops.aten._softmax_backward_data(dY, Y, -1, Y.dtype)


def bwd_ref64(Y, dY):
    Y, dY = Y.double(), dY.double()
    return Y * (dY - (Y * dY).sum(dim=-1, keepdim=True))


def bwd_sum_size(Y, dY):
    # dX_i = Y_i (dY_i - S): S's rounding error reaches dX_i times Y_i
    Y, dY = Y.double(), dY.double()
    return Y.abs() * (Y * dY).abs().sum(dim=-1, keepdim=True)


def elementwise(e: ElementwiseOp) -> Op:
    """The harness op for an entry of sol_kernels.elementwise_ops. Input i of a case comes
    from cases.make_input; a bool input is that tensor > 0. torch, run on meta tensors,
    gives the output's shape, strides and promoted dtype."""

    def make_inputs(case: Case, gen: torch.Generator):
        xs = [make_input(case, i, gen) for i in range(e.arity)]
        return tuple(x > 0 if i in e.bool_inputs else x for i, x in enumerate(xs))

    def out_meta(*xs):
        return e.torch_fn(*(x.to("meta") for x in xs))

    def build(*xs):
        out = out_meta(*xs)
        # torch rounds every input to the common dtype before computing. That is the
        # output dtype, except for a comparison, whose output is bool.
        common = torch.result_type(*xs) if out.dtype == torch.bool else out.dtype
        return elementwise_builder(e.f, *xs, out_like=out, common_dtype=common)

    def ref64(*xs):
        return e.torch_fn(*(x.double() if x.is_floating_point() else x for x in xs))

    def torch_accepts(case):  # torch.lerp, for one, rejects mixed dtypes
        try:
            out_meta(*make_inputs(replace(case, rows=2, cols=3), torch.Generator()))
        except RuntimeError:
            return False
        return True

    perf = Case(PERF_ELEMENTS // 4096, 4096, torch.bfloat16)
    perf_cases = [perf, replace(perf, layouts=("offset",) * e.arity)]
    if e.arity > 1:
        perf_cases.append(replace(perf, layouts=("contiguous", "row", "contiguous")[:e.arity]))

    return Op(e.name, build, make_inputs, e.torch_fn, e.torch_fn, ref64,
              nbytes=lambda *xs: sum(x.nbytes for x in xs) + out_meta(*xs).nbytes,
              make_out=lambda *xs: torch.empty_like(out_meta(*xs), device=xs[0].device),
              rule="exact" if e.exact else "relative",
              dev_cases=lambda: [c for c in elementwise_dev_cases(e.arity) if torch_accepts(c)],
              heldout_cases=lambda: [c for c in elementwise_heldout_cases(e.arity) if torch_accepts(c)],
              perf_cases=perf_cases)


def as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


def reduced_dims(x, dim) -> tuple:
    """dim as a sorted tuple of non-negative dims; None is every dim."""
    if dim is None:
        return tuple(range(x.dim()))
    return tuple(sorted(d % max(x.dim(), 1) for d in as_tuple(dim)))


def reduced_count(x, dim) -> int:
    return prod(x.shape[d] for d in reduced_dims(x, dim))


# The size of the summed terms that each float output of a reduction depends on, in float64
# and in the output's units (see docs/eval-design.md): an output may carry a float32 sum's
# rounding error of terms of that size. Each is (x, dim, keepdim, **options) -> one tensor per
# output. The other reductions pick an element of the input or follow from comparisons alone,
# so their outputs must equal torch's.
def mean_size(x, dim, keepdim):
    return x.abs().mean(reduced_dims(x, dim), keepdim=keepdim)


def var_size(x, dim, keepdim, correction):
    """var = sum((x - m)^2) / (n - correction). Besides the squares, every term (x - m) carries
    the error of the mean m, whose sum size is mean(|x|)."""
    dims = reduced_dims(x, dim)
    dev = x - x.mean(dims, keepdim=True)
    size = dev.square().sum(dims, keepdim=keepdim) + 2 * mean_size(x, dim, keepdim) * dev.abs().sum(dims, keepdim=keepdim)
    return size / max(reduced_count(x, dim) - correction, 0)


def std_size(x, dim, keepdim, correction):  # std = sqrt(var), so an error d in var is d / (2 std) in std
    std = x.var(reduced_dims(x, dim), correction=correction, keepdim=keepdim).sqrt()
    return torch.where(std > 0, var_size(x, dim, keepdim, correction) / (2 * std), 0.0)


def norm2_size(x, dim, keepdim):  # the norm is sqrt(s) for s = sum(x^2), of size s
    return torch.linalg.vector_norm(x, 2, reduced_dims(x, dim), keepdim) / 2


REDUCTION_SUM_SIZES = {
    "sum": lambda x, dim, keepdim: x.abs().sum(reduced_dims(x, dim), keepdim=keepdim),
    "mean": mean_size,
    "var": var_size,
    "std": std_size,
    "var_mean": lambda x, dim, keepdim, correction: (var_size(x, dim, keepdim, correction), mean_size(x, dim, keepdim)),
    "std_mean": lambda x, dim, keepdim, correction: (std_size(x, dim, keepdim, correction), mean_size(x, dim, keepdim)),
    "norm1": lambda x, dim, keepdim: torch.linalg.vector_norm(x, 1, reduced_dims(x, dim), keepdim),
    "norm2": norm2_size,
    # log(s) for s = sum(exp(x - max)), so an error of s's size moves the result by 1
    "logsumexp": lambda x, dim, keepdim: torch.ones_like(torch.logsumexp(x, reduced_dims(x, dim), keepdim)),
}

# torch's var_mean on CPU takes the mean from Welford's update along some dims, so it gives NaN
# for a mean over an inf; the reference takes it from the sum
def var_mean_ref(sqrt):
    def ref(x, dim, keepdim, correction):
        dims = reduced_dims(x, dim)
        var = x.var(dims, correction=correction, keepdim=keepdim)
        return var.sqrt() if sqrt else var, x.mean(dims, keepdim=keepdim)
    return ref


REDUCTION_REFERENCES = {"var_mean": var_mean_ref(False), "std_mean": var_mean_ref(True)}

# bf16 at 2^27 elements: rows, a few very long rows (the split path), columns, and all of it
REDUCTION_PERF_CASES = [ReductionCase((2**15, 2**12), -1, torch.bfloat16), ReductionCase((2**4, 2**23), -1, torch.bfloat16),
                        ReductionCase((2**12, 2**15), 0, torch.bfloat16), ReductionCase((2**27,), None, torch.bfloat16)]


def reduction(r: ReductionOp) -> Op:
    """The harness op for an entry of sol_kernels.reduction_ops. torch, run on meta tensors,
    gives the outputs' shapes, strides and dtypes."""

    def options(case):
        return {"dim": case.dim, "keepdim": case.keepdim} | {name: getattr(case, name) for name in r.options}

    def out_meta(x, **o):
        return as_tuple(r.torch_call(x.to("meta"), **o))

    def build(x, **o):
        return reduction_builder(r, x, dims=reduced_dims(x, o["dim"]), outs_like=out_meta(x, **o), params=r.params(**o))

    def torch_fn(x, *, out, **o):  # torch has no out= form of some of these, so this copies its result
        for y, t in zip(out, as_tuple(r.torch_call(x, **o))):
            y.copy_(t)

    def ref64(x, **o):
        return as_tuple(REDUCTION_REFERENCES.get(r.name, r.torch_call)(x.double() if x.is_floating_point() else x, **o))

    def sum_size(x, **o):
        return as_tuple(REDUCTION_SUM_SIZES[r.name](x.double(), **o))

    def make_out(x, **o):
        return tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=x.device) for t in out_meta(x, **o))

    def wanted(case):
        """Whether the op is checked on this case: torch must accept it (max takes one dim, amax
        no empty reduction), and only var and std take a correction."""
        if case.correction != 1 and "correction" not in r.options:
            return False
        small = replace(case, shape=tuple(min(n, 2) for n in case.shape), layout="contiguous")
        try:
            r.torch_call(make_reduction_input(small, torch.Generator()), **options(small))
            out_meta(torch.empty(case.shape, device="meta"), **options(case))  # some checks see the shape
        except (RuntimeError, IndexError, TypeError):
            return False
        return True

    n_out = len(out_meta(torch.empty(2, 2), dim=-1, keepdim=False, **{name: 1 for name in r.options}))
    rule = ("sum_scaled" if r.name in REDUCTION_SUM_SIZES else "exact",) * n_out
    return Op(r.name, build, lambda case, gen: (make_reduction_input(case, gen),), torch_fn, r.torch_call, ref64,
              nbytes=lambda x, **o: x.nbytes + sum(t.nbytes for t in out_meta(x, **o)), make_out=make_out, rule=rule,
              dev_cases=lambda: [c for c in reduction_dev_cases(r.bool_input) if wanted(c)],
              heldout_cases=lambda: [c for c in reduction_heldout_cases(r.bool_input) if wanted(c)],
              perf_cases=[c for c in REDUCTION_PERF_CASES if wanted(c)],
              sum_size=sum_size if r.name in REDUCTION_SUM_SIZES else None, options=options)


# layer_norm and rms_norm over the last dim of the softmax cases, with a weight (and a bias)
# of the row's length. They return the statistics their backward reads.
NORM_EPS = 1e-5
NORM_PERF_CASES = [Case(PERF_ELEMENTS // n, n, torch.bfloat16) for n in (128, 1024, 4096, 8192, 32768)]


def norm_inputs(case: Case, gen: torch.Generator, n_params: int):
    params = [make_tensor(replace(case, rows=1, offset=0, values="normal"), gen)[0] for _ in range(n_params)]
    return make_tensor(case, gen), *params


def layer_norm_torch(x, w, b):
    return torch.native_layer_norm(x, x.shape[-1:], w, b, NORM_EPS)


def layer_norm_sum_size(x, w, b):
    """out = (x - m) r w + b for r = (v + eps)^-1/2. An error d in the mean m moves out by
    |w| r d; an error d in the variance v moves r by r d / (2 (v + eps)), and so out - b by
    (out - b) d / (2 (v + eps)). The saved mean and r carry the same errors."""
    x, w, b = x.double(), w.double(), b.double()
    n = x.shape[-1]
    m = x.mean(-1, keepdim=True)
    mean_size = x.abs().mean(-1, keepdim=True)
    var = (x - m).square().mean(-1, keepdim=True)
    var_size = ((x - m).square().sum(-1, keepdim=True) + 2 * mean_size * (x - m).abs().sum(-1, keepdim=True)) / n
    r = (var + NORM_EPS).rsqrt()
    out = (x - m) * r * w + b
    return w.abs() * r * mean_size + (out - b).abs() * var_size / (2 * (var + NORM_EPS)), mean_size, r * var_size / (2 * (var + NORM_EPS))


def rms_norm_torch(x, w):
    return torch.ops.aten._fused_rms_norm(x, x.shape[-1:], w, NORM_EPS)


def rms_norm_sum_size(x, w):
    """out = x r w for r = (s + eps)^-1/2 and s = mean(x^2), whose terms are positive."""
    x, w = x.double(), w.double()
    s = x.square().mean(-1, keepdim=True)
    r = (s + NORM_EPS).rsqrt()
    return (x * r * w).abs() * s / (2 * (s + NORM_EPS)), r * s / (2 * (s + NORM_EPS))


def norm(name, fused, torch_call, n_params, sum_size) -> Op:
    def out_meta(*xs):
        return torch_call(*(x.to("meta") for x in xs))

    def torch_fn(*xs, out):
        for y, t in zip(out, torch_call(*xs)):
            y.copy_(t)

    def make_out(*xs):
        return tuple(torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=xs[0].device) for t in out_meta(*xs))

    def build(x, *params):
        return reduction_builder(fused, x, *params, dims=(x.dim() - 1,), outs_like=out_meta(x, *params), params=(NORM_EPS,))

    def sizes(*xs):  # 0 where the row's statistics overflow: then out is x * 0 or NaN
        return tuple(t.nan_to_num(0.0, 0.0, 0.0) for t in sum_size(*xs))

    return Op(name, build, lambda case, gen: norm_inputs(case, gen, n_params), torch_fn, torch_call,
              lambda *xs: torch_call(*(x.double() for x in xs)),
              nbytes=lambda *xs: sum(x.nbytes for x in xs) + sum(t.nbytes for t in out_meta(*xs)), make_out=make_out,
              rule=("sum_scaled",) * len(out_meta(*norm_inputs(Case(2, 2, torch.float32), torch.Generator(), n_params))),
              dev_cases=dev_cases, heldout_cases=heldout_cases, perf_cases=NORM_PERF_CASES, sum_size=sizes)


OPS = {
    "softmax_fwd": Op("softmax_fwd", softmax_fwd_builder, fwd_inputs, fwd_torch, fwd_torch_call, fwd_ref64,
                      nbytes=lambda X: 2 * X.nbytes, make_out=torch.empty_like, rule="max_scaled",
                      dev_cases=dev_cases, heldout_cases=heldout_cases, perf_cases=SOFTMAX_PERF_CASES),
    "softmax_bwd": Op("softmax_bwd", softmax_bwd_builder, bwd_inputs, bwd_torch, bwd_torch_call, bwd_ref64,
                      nbytes=lambda Y, dY: 3 * Y.nbytes, make_out=lambda Y, dY: torch.empty_like(Y),
                      rule="sum_scaled", dev_cases=dev_cases, heldout_cases=heldout_cases,
                      perf_cases=SOFTMAX_PERF_CASES, sum_size=bwd_sum_size),
    "layer_norm": norm("layer_norm", LAYER_NORM, layer_norm_torch, 2, layer_norm_sum_size),
    "rms_norm": norm("rms_norm", RMS_NORM, rms_norm_torch, 1, rms_norm_sum_size),
} | {e.name: elementwise(e) for e in ELEMENTWISE_OPS} | {r.name: reduction(r) for r in REDUCTION_OPS}
