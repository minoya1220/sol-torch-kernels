"""Torch elementwise ops written as per-element functions for elementwise_builder.

Each entry is the op's name, its arity, the function the kernel applies, and the torch call
the kernel must match, which accepts out=. The function takes one value per input in the
type torch computes in (see sol_kernels.elementwise), so one function serves every dtype.
sol_eval.ops turns each entry into an op the harness checks and times. Where torch spells
out its formula (lerp, hardswish, gelu and the backward ops), the function follows torch's
CUDA kernel step for step, constants included.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import cutlass
import cutlass.cute as cute
import torch
import torch.nn.functional as F
from cutlass import Boolean

aten = torch.ops.aten


@dataclass(frozen=True)
class ElementwiseOp:
    name: str
    arity: int
    f: Callable
    torch_fn: Callable
    bool_inputs: tuple = ()  # positions of inputs that are torch.bool, like where's condition
    # f takes the same correctly rounded steps as torch, so its result must equal torch's
    exact: bool = False
    # checked on integer and bool inputs as well as floats, where torch accepts them
    int_inputs: bool = True


# The values of C's M_SQRT2, M_2_SQRTPI and M_SQRT1_2, which torch's kernels use. Products of
# them are taken in float64, as in C, and rounded to the compute type where they are used.
SQRT2 = 1.41421356237309504880
TWO_OVER_SQRTPI = 1.12837916709551257390
SQRT1_2 = 0.70710678118654752440
# torch writes 1.0f / 6.0f, a float32, even where it computes in float64
ONE_SIXTH = (torch.tensor(1.0) / 6.0).item()
NAN = float("nan")
INF = float("inf")
# torch.nn.functional.selu's constants, which it passes to elu
SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


def is_float(x):
    return type(x).is_float


def select(cond, a, b):
    """cond ? a : b, in the type of whichever of a and b is a DSL value (the other may be a
    Python number)."""
    t = type(a) if isinstance(a, cutlass.Numeric) else type(b)
    return t(cutlass.select_(cond, t(a), t(b)))


# torch's maximum, minimum and clamp return NaN if any operand is NaN
def maximum(a, b):
    b = type(a)(b)
    return cute.math.max(a, b, propagate_nan=True) if is_float(a) else cute.math.max(a, b)


def minimum(a, b):
    b = type(a)(b)
    return cute.math.min(a, b, propagate_nan=True) if is_float(a) else cute.math.min(a, b)


def float_only(fn):
    """fn on floats; torch returns an integer unchanged (floor, ceil, trunc, round)."""
    return lambda a: fn(a) if is_float(a) else a


def sign(a):
    # c10::signum is (0 < a) - (a < 0), so NaN gives 0. A bool is its own sign.
    t = type(a)
    if t is Boolean:
        return a
    return select(a > 0, t(1), select(a < 0, -t(1), t(0)))


def isinf(a):
    return cute.math.abs(a) == INF if is_float(a) else Boolean(False)


def isfinite(a):
    return cute.math.abs(a) < INF if is_float(a) else Boolean(True)


def nan_to_num_finite(a):
    t = type(a)
    if not t.is_float:
        return a
    return select(a != a, t(0.0), select(isinf(a), select(a > 0.0, t(1.0), t(-1.0)), a))


def hypot(a, b):
    # big * sqrt(1 + (small / big)^2) does not overflow where a * a does; inf wins over NaN
    x, y = cute.math.abs(a), cute.math.abs(b)
    big, small = maximum(x, y), minimum(x, y)
    r = small / big
    h = select(big == 0.0, 0.0, big * cute.math.sqrt(1.0 + r * r))
    return select((x == INF) | (y == INF), INF, h)


def lerp(start, end, weight):
    diff = end - start
    return select(cute.math.abs(weight) < 0.5, start + weight * diff, end - diff * (1.0 - weight))


def remainder(a, b):
    # the sign of the result follows b, as in Python
    mod = cute.math.rem(a, b)
    return select((mod != 0.0) & ((b < 0.0) != (mod < 0.0)), mod + b, mod)


def left_shift(a, b):
    # torch: a shift by a negative amount, or by the width or more, gives 0
    t = type(a)
    return select((b < 0) | (b >= t.width), t(0), t(a << b))


def right_shift(a, b):
    # torch: out of range, a signed value keeps only its sign and an unsigned one becomes 0
    t = type(a)
    if t.signed:
        return select((b < 0) | (b >= t.width - 1), t(a >> (t.width - 1)), t(a >> b))
    return select(b >= t.width, t(0), t(a >> b))


def gelu(x):
    return x * 0.5 * (1.0 + cute.math.erf(x * SQRT1_2))


def gelu_tanh(x):
    x_cube = x * x * x
    inner = SQRT2 * TWO_OVER_SQRTPI * 0.5 * (x + 0.044715 * x_cube)
    return 0.5 * x * (1.0 + cute.math.tanh(inner))


def elu(a, alpha=1.0, scale=1.0, input_scale=1.0):
    t = type(a)
    negcoef = t(alpha) * t(scale)  # torch multiplies them in the compute type
    return select(a > 0.0, a * t(scale), cute.math.expm1(a * t(input_scale)) * negcoef)


def softplus(a, beta=1.0, threshold=20.0):
    return select(a * beta > threshold, a, cute.math.log1p(cute.math.exp(a * beta)) / beta)


def gelu_backward(dy, x):
    beta = TWO_OVER_SQRTPI * SQRT1_2 * 0.5
    cdf = 0.5 * (1.0 + cute.math.erf(x * SQRT1_2))
    pdf = cute.math.exp(-0.5 * x * x) * beta
    return dy * (cdf + x * pdf)


def gelu_tanh_backward(dy, x):
    beta = SQRT2 * TWO_OVER_SQRTPI * 0.5
    kappa = 0.044715
    x_sq = x * x
    x_cube = x_sq * x
    inner = beta * (x + kappa * x_cube)
    tanh_inner = cute.math.tanh(inner)
    left = 0.5 * x
    right = 1.0 + tanh_inner
    left_derivative = 0.5 * right
    tanh_derivative = 1.0 - tanh_inner * tanh_inner
    inner_derivative = beta * (1.0 + 3.0 * kappa * x_sq)
    right_derivative = left * tanh_derivative * inner_derivative
    return dy * (left_derivative + right_derivative)


def silu_backward(dy, x):
    s = 1.0 / (1.0 + cute.math.exp(-x))
    return dy * s * (1.0 + x * (1.0 - s))


def mish_backward(dy, x):
    s = 1.0 / (1.0 + cute.math.exp(-x))
    t = cute.math.tanh(cute.math.log1p(cute.math.exp(x)))
    return dy * (t + x * s * (1.0 - t * t))


def softplus_backward(dy, x, beta=1.0, threshold=20.0):
    z = cute.math.exp(x * beta)
    return select(x * beta > threshold, dy, dy * z / (z + 1.0))


def elu_backward(dy, x, alpha=1.0, scale=1.0, input_scale=1.0):
    t = type(x)
    negcoef = t(alpha) * t(scale)
    return select(x <= 0.0, dy * t(input_scale) * negcoef * cute.math.exp(x * t(input_scale)), dy * t(scale))


def hardswish_backward(dy, x):
    return select(x <= -3.0, 0.0, select(x < 3.0, dy * ((x / 3.0) + 0.5), dy))


def grad_input(fn):
    """torch's backward ops name their out= argument grad_input=."""
    return lambda *xs, out=None: fn(*xs) if out is None else fn(*xs, grad_input=out)


def copy_out(fn):
    """For a torch op with no out= form: the checks copy its result into out."""
    return lambda *xs, out=None: fn(*xs) if out is None else out.copy_(fn(*xs))


def to(dtype):
    """Tensor.to(dtype). The out= form of _to_copy takes the dtype from out."""
    return lambda x, out=None: aten._to_copy(x, dtype=dtype) if out is None else aten._to_copy.out(x, out=out)


CAST_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64, torch.uint8, torch.int8,
               torch.int16, torch.int32, torch.int64, torch.bool)

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
    # On integers, fmod and remainder by zero and pow to a negative power have no defined
    # result in torch's CUDA kernels, and raise on the CPU, so they are checked on floats only.
    ElementwiseOp("fmod", 2, lambda a, b: cute.math.rem(a, b), torch.fmod, exact=True, int_inputs=False),
    ElementwiseOp("remainder", 2, remainder, torch.remainder, exact=True, int_inputs=False),
    ElementwiseOp("pow", 2, lambda a, b: cute.math.pow(a, b), torch.pow, int_inputs=False),
    ElementwiseOp("maximum", 2, maximum, torch.maximum, exact=True),
    ElementwiseOp("minimum", 2, minimum, torch.minimum, exact=True),
    ElementwiseOp("atan2", 2, lambda a, b: cute.math.atan2(a, b), torch.atan2),
    ElementwiseOp("hypot", 2, hypot, torch.hypot),
    ElementwiseOp("copysign", 2, lambda a, b: cute.math.copysign(a, b), torch.copysign, exact=True),
    ElementwiseOp("addcmul", 3, lambda x, a, b: x + a * b, torch.addcmul),
    ElementwiseOp("addcdiv", 3, lambda x, a, b: x + a / b, torch.addcdiv),
    ElementwiseOp("lerp", 3, lerp, torch.lerp),
    # rounding
    ElementwiseOp("floor", 1, float_only(cute.math.floor), torch.floor, exact=True),
    ElementwiseOp("ceil", 1, float_only(cute.math.ceil), torch.ceil, exact=True),
    ElementwiseOp("trunc", 1, float_only(cute.math.trunc), torch.trunc, exact=True),
    ElementwiseOp("round", 1, float_only(cute.math.roundeven), torch.round, exact=True),
    ElementwiseOp("sign", 1, sign, torch.sign, exact=True),
    # comparisons, tests and selection
    ElementwiseOp("eq", 2, lambda a, b: a == b, torch.eq, exact=True),
    ElementwiseOp("ne", 2, lambda a, b: a != b, torch.ne, exact=True),
    ElementwiseOp("lt", 2, lambda a, b: a < b, torch.lt, exact=True),
    ElementwiseOp("le", 2, lambda a, b: a <= b, torch.le, exact=True),
    ElementwiseOp("gt", 2, lambda a, b: a > b, torch.gt, exact=True),
    ElementwiseOp("ge", 2, lambda a, b: a >= b, torch.ge, exact=True),
    ElementwiseOp("isnan", 1, lambda a: a != a, torch.isnan, exact=True),
    ElementwiseOp("isinf", 1, isinf, torch.isinf, exact=True),
    ElementwiseOp("isfinite", 1, isfinite, copy_out(torch.isfinite), exact=True),
    # nan_to_num's default replaces inf with the largest value of the output dtype, which f
    # does not know when it computes a 16-bit output in float32; finite values stand in
    ElementwiseOp("nan_to_num_finite", 1, nan_to_num_finite,
                  partial(torch.nan_to_num, nan=0.0, posinf=1.0, neginf=-1.0), exact=True),
    ElementwiseOp("where", 3, lambda c, a, b: select(c != 0, a, b), torch.where, bool_inputs=(0,), exact=True),
    ElementwiseOp("clamp", 3, lambda x, lo, hi: minimum(maximum(x, lo), hi), torch.clamp, exact=True),
    ElementwiseOp("clamp_scalar", 1, lambda x: minimum(maximum(x, -0.5), 0.5),
                  partial(torch.clamp, min=-0.5, max=0.5), exact=True),
    # bitwise, on integers and bools
    ElementwiseOp("bitwise_and", 2, lambda a, b: a & b, torch.bitwise_and, exact=True),
    ElementwiseOp("bitwise_or", 2, lambda a, b: a | b, torch.bitwise_or, exact=True),
    ElementwiseOp("bitwise_xor", 2, lambda a, b: a ^ b, torch.bitwise_xor, exact=True),
    ElementwiseOp("bitwise_not", 1, lambda a: ~a, torch.bitwise_not, exact=True),
    ElementwiseOp("bitwise_left_shift", 2, left_shift, torch.bitwise_left_shift, exact=True),
    ElementwiseOp("bitwise_right_shift", 2, right_shift, torch.bitwise_right_shift, exact=True),
    # casts: f is the identity, because the builder's conversion to the common dtype, which
    # is the target, is the cast
    *[ElementwiseOp("to_" + str(d).removeprefix("torch."), 1, lambda a: a, to(d), exact=True) for d in CAST_DTYPES],
    # activations
    ElementwiseOp("relu", 1, lambda a: maximum(a, 0), aten.relu, exact=True),
    ElementwiseOp("leaky_relu", 1, lambda a: select(a > 0.0, a, a * 0.01), aten.leaky_relu, exact=True),
    ElementwiseOp("leaky_relu_0.2", 1, lambda a: select(a > 0.0, a, a * 0.2),
                  partial(aten.leaky_relu, negative_slope=0.2), exact=True),
    ElementwiseOp("elu", 1, elu, aten.elu),
    ElementwiseOp("elu_alpha_0.5", 1, partial(elu, alpha=0.5), partial(aten.elu, alpha=0.5)),
    ElementwiseOp("selu", 1, partial(elu, alpha=SELU_ALPHA, scale=SELU_SCALE), copy_out(aten.selu)),
    ElementwiseOp("celu", 1, elu, aten.celu),  # celu(alpha=1) is elu with input_scale 1 / alpha = 1
    ElementwiseOp("sigmoid", 1, lambda a: 1.0 / (1.0 + cute.math.exp(-a)), torch.sigmoid),
    ElementwiseOp("logit", 1, lambda a: cute.math.log(a / (1.0 - a)), torch.logit),
    ElementwiseOp("silu", 1, lambda a: a / (1.0 + cute.math.exp(-a)), aten.silu),
    ElementwiseOp("gelu", 1, gelu, aten.gelu),
    ElementwiseOp("gelu_tanh", 1, gelu_tanh, partial(aten.gelu, approximate="tanh")),
    ElementwiseOp("softplus", 1, softplus, aten.softplus),
    ElementwiseOp("softplus_beta_2_threshold_10", 1, partial(softplus, beta=2.0, threshold=10.0),
                  partial(aten.softplus, beta=2.0, threshold=10.0)),
    # torch's softsign is x / (x.abs() + 1) in three ops, so on integers abs and + wrap in the
    # integer type before the division
    ElementwiseOp("softsign", 1, lambda a: a / (cute.math.abs(a) + 1.0), copy_out(F.softsign), int_inputs=False),
    ElementwiseOp("mish", 1, lambda a: a * cute.math.tanh(cute.math.log1p(cute.math.exp(a))), aten.mish),
    ElementwiseOp("hardsigmoid", 1, lambda a: minimum(maximum(a + 3.0, 0.0), 6.0) * ONE_SIXTH, aten.hardsigmoid),
    ElementwiseOp("hardswish", 1, lambda a: a * minimum(maximum(a + 3.0, 0.0), 6.0) * ONE_SIXTH, aten.hardswish),
    # backward ops, in the form torch's autograd calls them: the gradient first, then the
    # forward's input (its output for threshold, sigmoid and tanh)
    ElementwiseOp("threshold_backward", 2, lambda dy, y: select(y <= 0, 0, dy),
                  grad_input(partial(aten.threshold_backward, threshold=0)), exact=True),
    ElementwiseOp("sigmoid_backward", 2, lambda dy, y: dy * (1.0 - y) * y, grad_input(aten.sigmoid_backward)),
    ElementwiseOp("tanh_backward", 2, lambda dy, y: dy * (1.0 - y * y), grad_input(aten.tanh_backward)),
    ElementwiseOp("gelu_backward", 2, gelu_backward, grad_input(aten.gelu_backward)),
    ElementwiseOp("gelu_tanh_backward", 2, gelu_tanh_backward,
                  grad_input(partial(aten.gelu_backward, approximate="tanh"))),
    ElementwiseOp("silu_backward", 2, silu_backward, grad_input(aten.silu_backward)),
    ElementwiseOp("mish_backward", 2, mish_backward, copy_out(aten.mish_backward)),
    ElementwiseOp("softplus_backward", 2, softplus_backward,
                  grad_input(partial(aten.softplus_backward, beta=1.0, threshold=20.0))),
    ElementwiseOp("elu_backward", 2, elu_backward,
                  grad_input(lambda dy, x, **out: aten.elu_backward(dy, 1.0, 1.0, 1.0, False, x, **out))),
    ElementwiseOp("hardsigmoid_backward", 2, lambda dy, x: select((x > -3.0) & (x < 3.0), dy * ONE_SIXTH, 0.0),
                  grad_input(aten.hardsigmoid_backward), exact=True),
    ElementwiseOp("hardswish_backward", 2, hardswish_backward, aten.hardswish_backward),
    ElementwiseOp("leaky_relu_backward", 2, lambda dy, x: select(x > 0.0, dy, dy * 0.01),
                  grad_input(partial(aten.leaky_relu_backward, negative_slope=0.01, self_is_result=False)),
                  exact=True),
    ElementwiseOp("hardtanh_backward", 2, lambda dy, x: select((x <= -1.0) | (x >= 1.0), 0.0, dy),
                  grad_input(partial(aten.hardtanh_backward, min_val=-1.0, max_val=1.0)), exact=True),
    ElementwiseOp("hardshrink_backward", 2, lambda dy, x: select((x >= -0.5) & (x <= 0.5), 0.0, dy),
                  grad_input(partial(aten.hardshrink_backward, lambd=0.5)), exact=True),
    ElementwiseOp("softshrink_backward", 2, lambda dy, x: select((x >= -0.5) & (x <= 0.5), 0.0, dy),
                  grad_input(partial(aten.softshrink_backward, lambd=0.5)), exact=True),
    ElementwiseOp("logit_backward", 2, lambda dy, x: select((x < 0.0) | (x > 1.0), NAN, dy / (x * (1.0 - x))),
                  grad_input(aten.logit_backward)),
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
    ElementwiseOp("asin", 1, lambda a: cute.math.asin(a), torch.asin),
    ElementwiseOp("acos", 1, lambda a: cute.math.acos(a), torch.acos),
    ElementwiseOp("atan", 1, lambda a: cute.math.atan(a), torch.atan),
    ElementwiseOp("sinh", 1, lambda a: cute.math.sinh(a), torch.sinh),
    ElementwiseOp("cosh", 1, lambda a: cute.math.cosh(a), torch.cosh),
    ElementwiseOp("tanh", 1, lambda a: cute.math.tanh(a), torch.tanh),
    ElementwiseOp("asinh", 1, lambda a: cute.math.asinh(a), torch.asinh),
    ElementwiseOp("acosh", 1, lambda a: cute.math.acosh(a), torch.acosh),
    ElementwiseOp("atanh", 1, lambda a: cute.math.atanh(a), torch.atanh),
    ElementwiseOp("erf", 1, lambda a: cute.math.erf(a), torch.erf),
    ElementwiseOp("erfc", 1, lambda a: cute.math.erfc(a), torch.erfc),
]
