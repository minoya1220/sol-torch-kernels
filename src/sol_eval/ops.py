"""The ops the harness can check and time. Each says how to make inputs for a case,
how torch computes it in the kernel's call form and in its usual call, its float64
reference, the bytes it must move, how to allocate its output, which error rule it is
checked with, and its cases."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from sol_eval.cases import (Case, dev_cases, elementwise_dev_cases, elementwise_heldout_cases, heldout_cases,
                            make_input, make_tensor)
from sol_kernels.elementwise import elementwise_builder
from sol_kernels.elementwise_ops import ELEMENTWISE_OPS, ElementwiseOp
from sol_kernels.softmax_bwd import softmax_bwd_builder
from sol_kernels.softmax_fwd import softmax_fwd_builder


@dataclass(frozen=True)
class Op:
    name: str
    build: Callable        # build(*inputs) -> fn, and fn(*inputs, out=) writes out
    make_inputs: Callable  # (case, generator) -> tuple of input tensors
    torch_fn: Callable     # torch in the same call form as fn, for the checks
    torch_call: Callable   # torch's usual call, without out=, which is what is timed
    ref64: Callable        # (*inputs) -> float64 result
    make_out: Callable     # (*inputs) -> an empty output for fn to write
    rule: str              # the error rule in sol_eval.check
    dev_cases: Callable    # () -> correctness cases
    heldout_cases: Callable
    perf_cases: list       # cases to time
    sum_size: Callable | None = None  # (*inputs) -> per element, the size of the terms of the sum it uses
    # Exactly one of these says what the op's speed of light is measured in.
    nbytes: Callable | None = None  # memory-bound: (*inputs) -> bytes the op must read and write
    flops: Callable | None = None   # compute-bound: (*inputs, **options) -> FLOPs the op must do
    # Keyword arguments of the call that are not tensors, such as attention's is_causal and
    # scale. They go to build, torch_fn, torch_call, ref64, sum_size and flops.
    options: Callable = lambda case: {}
    torch_backend: Callable | None = None  # (*inputs, **options) -> which of torch's kernels its call runs


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


OPS = {
    "softmax_fwd": Op("softmax_fwd", softmax_fwd_builder, fwd_inputs, fwd_torch, fwd_torch_call, fwd_ref64,
                      nbytes=lambda X: 2 * X.nbytes, make_out=torch.empty_like, rule="max_scaled",
                      dev_cases=dev_cases, heldout_cases=heldout_cases, perf_cases=SOFTMAX_PERF_CASES),
    "softmax_bwd": Op("softmax_bwd", softmax_bwd_builder, bwd_inputs, bwd_torch, bwd_torch_call, bwd_ref64,
                      nbytes=lambda Y, dY: 3 * Y.nbytes, make_out=lambda Y, dY: torch.empty_like(Y),
                      rule="sum_scaled", dev_cases=dev_cases, heldout_cases=heldout_cases,
                      perf_cases=SOFTMAX_PERF_CASES, sum_size=bwd_sum_size),
} | {e.name: elementwise(e) for e in ELEMENTWISE_OPS}
