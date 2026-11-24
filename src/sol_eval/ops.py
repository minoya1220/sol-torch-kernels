"""The ops the harness can check and time. Each says how to make inputs for a case,
how torch computes it in the kernel's call form, its float64 reference, the bytes it
must move, and how to allocate its output."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from sol_eval.cases import Case, make_tensor
from sol_kernels.elementwise import elementwise_builder
from sol_kernels.elementwise_ops import ELEMENTWISE_OPS, ElementwiseOp
from sol_kernels.softmax_bwd import softmax_bwd_builder
from sol_kernels.softmax_fwd import softmax_fwd_builder


@dataclass(frozen=True)
class Op:
    name: str
    build: Callable        # build(*inputs) -> fn, and fn(*inputs, out=) writes out
    make_inputs: Callable  # (case, generator) -> tuple of input tensors
    torch_fn: Callable     # torch in the same call form as fn
    ref64: Callable        # (*inputs) -> float64 result
    nbytes: Callable       # (*inputs) -> bytes the op must read and write
    make_out: Callable     # (*inputs) -> an empty output for fn to write


def fwd_inputs(case: Case, gen: torch.Generator):
    return (make_tensor(case, gen),)


def fwd_torch(X, *, out):
    return torch.ops.aten._softmax.out(X, -1, False, out=out)


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


def bwd_ref64(Y, dY):
    Y, dY = Y.double(), dY.double()
    return Y * (dY - (Y * dY).sum(dim=-1, keepdim=True))


def elementwise(e: ElementwiseOp) -> Op:
    """The harness op for an entry of sol_kernels.elementwise_ops. Every input is a tensor
    of the case's shape, dtype, offset and kind of values; a bool input is one of them > 0.
    torch, run on meta tensors, gives the output's shape and promoted dtype."""

    def make_inputs(case: Case, gen: torch.Generator):
        xs = [make_tensor(case, gen) for _ in range(e.arity)]
        return tuple(x > 0 if i in e.bool_inputs else x for i, x in enumerate(xs))

    def out_meta(*xs):
        return e.torch_fn(*(x.to("meta") for x in xs))

    def build(*xs):
        return elementwise_builder(e.f, *xs, out_dtype=out_meta(*xs).dtype)

    def ref64(*xs):
        return e.torch_fn(*(x.double() if x.is_floating_point() else x for x in xs))

    return Op(e.name, build, make_inputs, e.torch_fn, ref64,
              nbytes=lambda *xs: sum(x.nbytes for x in xs) + out_meta(*xs).nbytes,
              make_out=lambda *xs: torch.empty_like(out_meta(*xs), device=xs[0].device))


OPS = {
    "softmax_fwd": Op("softmax_fwd", softmax_fwd_builder, fwd_inputs, fwd_torch, fwd_ref64,
                      nbytes=lambda X: 2 * X.nbytes, make_out=torch.empty_like),
    "softmax_bwd": Op("softmax_bwd", softmax_bwd_builder, bwd_inputs, bwd_torch, bwd_ref64,
                      nbytes=lambda Y, dY: 3 * Y.nbytes, make_out=lambda Y, dY: torch.empty_like(Y)),
} | {e.name: elementwise(e) for e in ELEMENTWISE_OPS}
