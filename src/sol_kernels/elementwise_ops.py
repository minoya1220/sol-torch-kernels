"""Torch elementwise ops written as per-element functions for elementwise_builder.

Each entry is the op's name, its arity, the function of Float32 values the kernel applies,
and the torch call the kernel must match, which accepts out=. sol_eval.ops turns each entry
into an op the harness checks and times. Where torch spells out its formula (lerp,
hardswish, gelu), the function follows it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32

aten = torch.ops.aten


@dataclass(frozen=True)
class ElementwiseOp:
    name: str
    arity: int
    f: Callable
    torch_fn: Callable
    bool_inputs: tuple = ()  # positions of inputs that are torch.bool, like where's condition
    # f takes the same correctly rounded float32 steps as torch, so its result must be torch's, bit for bit
    exact: bool = False


def select(cond, a, b):
    return Float32(cutlass.select_(cond, Float32(a), Float32(b)))


# torch's maximum, minimum and clamp return NaN if any operand is NaN
def maximum(a, b):
    return cute.math.max(Float32(a), Float32(b), propagate_nan=True)


def minimum(a, b):
    return cute.math.min(Float32(a), Float32(b), propagate_nan=True)


def gelu(x):
    return x * 0.5 * (1.0 + cute.math.erf(x * 0.7071067811865476))


def gelu_tanh(x):
    x_cube = x * x * x
    inner = 0.7978845608028654 * (x + 0.044715 * x_cube)  # sqrt(2 / pi)
    return 0.5 * x * (1.0 + cute.math.tanh(inner))


def lerp(start, end, weight):
    diff = end - start
    return select(cute.math.abs(weight) < 0.5, start + weight * diff, end - diff * (1.0 - weight))


def remainder(a, b):
    # the sign of the result follows b, as in Python
    mod = cute.math.rem(a, b)
    return select((mod != 0.0) & ((b < 0.0) != (mod < 0.0)), mod + b, mod)


ELEMENTWISE_OPS = [
    # arithmetic
    ElementwiseOp("neg", 1, lambda a: -a, torch.neg, exact=True),
    ElementwiseOp("abs", 1, lambda a: cute.math.abs(a), torch.abs, exact=True),
    ElementwiseOp("square", 1, lambda a: a * a, torch.square, exact=True),
    ElementwiseOp("reciprocal", 1, lambda a: 1.0 / a, torch.reciprocal, exact=True),
    ElementwiseOp("add", 2, lambda a, b: a + b, torch.add, exact=True),
    ElementwiseOp("sub", 2, lambda a, b: a - b, torch.sub, exact=True),
    ElementwiseOp("mul", 2, lambda a, b: a * b, torch.mul, exact=True),
    ElementwiseOp("div", 2, lambda a, b: a / b, torch.div, exact=True),
    ElementwiseOp("fmod", 2, lambda a, b: cute.math.rem(a, b), torch.fmod, exact=True),
    ElementwiseOp("remainder", 2, remainder, torch.remainder, exact=True),
    ElementwiseOp("maximum", 2, maximum, torch.maximum, exact=True),
    ElementwiseOp("minimum", 2, minimum, torch.minimum, exact=True),
    ElementwiseOp("pow", 2, lambda a, b: cute.math.pow(a, b), torch.pow),
    ElementwiseOp("atan2", 2, lambda a, b: cute.math.atan2(a, b), torch.atan2),
    ElementwiseOp("addcmul", 3, lambda x, a, b: x + a * b, torch.addcmul),
    ElementwiseOp("addcdiv", 3, lambda x, a, b: x + a / b, torch.addcdiv),
    ElementwiseOp("lerp", 3, lerp, torch.lerp),
    # rounding
    ElementwiseOp("floor", 1, lambda a: cute.math.floor(a), torch.floor, exact=True),
    ElementwiseOp("ceil", 1, lambda a: cute.math.ceil(a), torch.ceil, exact=True),
    ElementwiseOp("trunc", 1, lambda a: cute.math.trunc(a), torch.trunc, exact=True),
    ElementwiseOp("round", 1, lambda a: cute.math.roundeven(a), torch.round, exact=True),
    # comparisons and selection
    ElementwiseOp("eq", 2, lambda a, b: a == b, torch.eq, exact=True),
    ElementwiseOp("ne", 2, lambda a, b: a != b, torch.ne, exact=True),
    ElementwiseOp("lt", 2, lambda a, b: a < b, torch.lt, exact=True),
    ElementwiseOp("le", 2, lambda a, b: a <= b, torch.le, exact=True),
    ElementwiseOp("gt", 2, lambda a, b: a > b, torch.gt, exact=True),
    ElementwiseOp("ge", 2, lambda a, b: a >= b, torch.ge, exact=True),
    ElementwiseOp("where", 3, lambda c, a, b: select(c != 0.0, a, b), torch.where, bool_inputs=(0,), exact=True),
    ElementwiseOp("clamp", 3, lambda x, lo, hi: minimum(maximum(x, lo), hi), torch.clamp, exact=True),
    # activations
    ElementwiseOp("relu", 1, lambda a: maximum(a, 0.0), aten.relu, exact=True),
    ElementwiseOp("leaky_relu", 1, lambda a: select(a > 0.0, a, a * 0.01), aten.leaky_relu, exact=True),
    ElementwiseOp("elu", 1, lambda a: select(a > 0.0, a, cute.math.expm1(a)), aten.elu),
    ElementwiseOp("sigmoid", 1, lambda a: 1.0 / (1.0 + cute.math.exp(-a)), torch.sigmoid),
    ElementwiseOp("silu", 1, lambda a: a / (1.0 + cute.math.exp(-a)), aten.silu),
    ElementwiseOp("gelu", 1, gelu, aten.gelu),
    ElementwiseOp("gelu_tanh", 1, gelu_tanh, partial(aten.gelu, approximate="tanh")),
    ElementwiseOp("softplus", 1, lambda a: select(a > 20.0, a, cute.math.log1p(cute.math.exp(a))), aten.softplus),
    ElementwiseOp("mish", 1, lambda a: a * cute.math.tanh(cute.math.log1p(cute.math.exp(a))), aten.mish),
    ElementwiseOp("hardsigmoid", 1, lambda a: minimum(maximum(a + 3.0, 0.0), 6.0) * (1.0 / 6.0), aten.hardsigmoid),
    ElementwiseOp("hardswish", 1, lambda a: a * minimum(maximum(a + 3.0, 0.0), 6.0) * (1.0 / 6.0), aten.hardswish),
    # math functions
    ElementwiseOp("exp", 1, lambda a: cute.math.exp(a), torch.exp),
    ElementwiseOp("exp2", 1, lambda a: cute.math.exp2(a), torch.exp2),
    ElementwiseOp("expm1", 1, lambda a: cute.math.expm1(a), torch.expm1),
    ElementwiseOp("log", 1, lambda a: cute.math.log(a), torch.log),
    ElementwiseOp("log2", 1, lambda a: cute.math.log2(a), torch.log2),
    ElementwiseOp("log10", 1, lambda a: cute.math.log10(a), torch.log10),
    ElementwiseOp("log1p", 1, lambda a: cute.math.log1p(a), torch.log1p),
    ElementwiseOp("sqrt", 1, lambda a: cute.math.sqrt(a), torch.sqrt, exact=True),
    ElementwiseOp("rsqrt", 1, lambda a: cute.math.rsqrt(a), torch.rsqrt),
    ElementwiseOp("sin", 1, lambda a: cute.math.sin(a), torch.sin),
    ElementwiseOp("cos", 1, lambda a: cute.math.cos(a), torch.cos),
    ElementwiseOp("tan", 1, lambda a: cute.math.tan(a), torch.tan),
    ElementwiseOp("tanh", 1, lambda a: cute.math.tanh(a), torch.tanh),
    ElementwiseOp("erf", 1, lambda a: cute.math.erf(a), torch.erf),
    ElementwiseOp("erfc", 1, lambda a: cute.math.erfc(a), torch.erfc),
]
