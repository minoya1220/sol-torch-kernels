"""Correctness cases. Both softmax ops take a 2-D row-major tensor, so a case is a
shape, a dtype, an offset into the allocation, a kind of values and a seed for them.
An elementwise op's inputs broadcast to the case's shape, and each can have its own layout
and dtype."""

import os
import random
from dataclasses import dataclass, replace
from math import prod

import torch

# float64 is left out: the kernels compute in float32 (see the softmax builders)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
DEV_ROW_LENGTHS = (3, 7, 8, 17, 64, 100, 257, 511, 633, 1021, 2049, 4099,
                   128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
EXTRA_ROW_LENGTHS = (7, 1021, 4099, 65536)
# how an elementwise input is laid out; each but "contiguous" takes a different path in the kernel
LAYOUTS = (
    "contiguous",
    "offset",      # one element into its allocation, so not 16-byte aligned
    "padded",      # rows 3 elements apart more than their length, so each row starts at a new alignment
    "transposed",  # stride 1 along the rows
    "strided",     # every other column
    "row",         # shape (cols,), broadcast across the rows
    "column",      # shape (rows, 1), broadcast along each row
    "scalar",      # a 0-dim float32 tensor
)
# input dtypes for cases that mix them: bf16 with fp16 computes in fp32
MIXED_DTYPES = ((torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16),
                (torch.bfloat16, torch.float32), (torch.float32, torch.float16))


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int
    dtype: torch.dtype
    offset: int = 0         # elements between the start of the allocation and the tensor
    values: str = "normal"  # "normal", "wide", "special" or "edge"
    seed: int = 0
    layouts: tuple = ()  # one per elementwise input; none means every input is contiguous
    dtypes: tuple = ()   # one per elementwise input; none means every input has `dtype`

    def __str__(self):
        name = f"{self.rows}x{self.cols}-{str(self.dtype).removeprefix('torch.')}-off{self.offset}-{self.values}"
        if self.layouts:
            name += "-" + ",".join(self.layouts)
        if self.dtypes:
            name += "-" + ",".join(str(d).removeprefix("torch.") for d in self.dtypes)
        return name


def dev_cases() -> list[Case]:
    cases = [Case(63, n, d) for n in DEV_ROW_LENGTHS for d in DTYPES]
    for n in EXTRA_ROW_LENGTHS:
        for d in DTYPES:
            cases.append(Case(63, n, d, offset=1))  # breaks 16-byte alignment
            cases.append(Case(63, n, d, values="wide"))
            cases.append(Case(63, n, d, values="special"))
    return cases


def heldout_cases(count: int = 100) -> list[Case]:
    """Cases from the same distribution family as the dev cases, drawn from a seed that is
    never committed, so an author cannot learn them."""
    seed = os.environ.get("SOL_HELDOUT_SEED")
    if not seed:
        raise RuntimeError("set SOL_HELDOUT_SEED to run the held-out cases")
    rng = random.Random(seed)
    cases = []
    for _ in range(count):
        cols = max(2, round(2 ** rng.uniform(1, 17)))
        rows = rng.randint(1, max(1, min(512, 2**22 // cols)))
        dtype = rng.choice(DTYPES)
        offset = rng.choice((0, 0, 1, 2, 3))
        values = rng.choice(("normal", "normal", "wide", "special"))
        cases.append(Case(rows, cols, dtype, offset, values, seed=rng.getrandbits(63)))
    return cases


def elementwise_dev_cases(arity: int) -> list[Case]:
    """The softmax dev cases, with every input contiguous, plus each layout and some mixed
    dtypes on one odd shape."""
    cases = dev_cases()
    for d in DTYPES:
        cases.append(Case(63, 1021, d, values="edge"))
        for layout in LAYOUTS[1:]:
            for i in range(arity):  # this input in the layout, the others contiguous
                cases.append(Case(63, 1021, d, layouts=tuple(layout if j == i else "contiguous" for j in range(arity))))
            if arity > 1:
                cases.append(Case(63, 1021, d, layouts=(layout,) * arity))
    if arity > 1:
        for a, b in MIXED_DTYPES:
            cases.append(Case(63, 1021, a, dtypes=(a, b, a)[:arity]))
    return cases


def elementwise_heldout_cases(arity: int, count: int = 100) -> list[Case]:
    """heldout_cases, with a layout and a dtype drawn for every input."""
    rng = random.Random(os.environ.get("SOL_HELDOUT_SEED", "") + f"-elementwise-{arity}")
    cases = []
    for case in heldout_cases(count):
        layouts = tuple(rng.choice(LAYOUTS[:1] * 3 + LAYOUTS[1:]) for _ in range(arity))
        dtypes = tuple(rng.choice((case.dtype,) * 3 + DTYPES) for _ in range(arity))
        values = rng.choice((case.values, "edge"))
        cases.append(replace(case, values=values, layouts=layouts, dtypes=dtypes))
    return cases


def make_input(case: Case, i: int, gen: torch.Generator) -> torch.Tensor:
    """Input i of an elementwise op: a tensor in input i's layout and dtype that broadcasts to
    the case's shape. Callers seed gen with case.seed."""
    layout = case.layouts[i] if case.layouts else "contiguous"
    c = replace(case, dtype=case.dtypes[i] if case.dtypes else case.dtype)
    rows, cols = case.rows, case.cols
    if layout == "contiguous":
        return make_tensor(c, gen)
    if layout == "offset":
        return make_tensor(replace(c, offset=1), gen)
    if layout == "padded":
        return make_tensor(replace(c, cols=cols + 3), gen)[:, :cols]
    if layout == "transposed":
        return make_tensor(replace(c, rows=cols, cols=rows), gen).t()
    if layout == "strided":
        return make_tensor(replace(c, cols=2 * cols), gen)[:, ::2]
    if layout == "row":
        return make_tensor(replace(c, rows=1), gen)[0]
    if layout == "column":
        return make_tensor(replace(c, cols=1), gen)
    if layout == "scalar":
        return make_tensor(replace(c, rows=1, cols=1, dtype=torch.float32), gen)[0, 0]
    raise ValueError(layout)


def make_tensor(case: Case, gen: torch.Generator) -> torch.Tensor:
    """A tensor for the case, on the generator's device. Callers seed gen with case.seed."""
    n = case.rows * case.cols
    base = torch.randn(case.offset + n, generator=gen, device=gen.device).to(case.dtype)
    x = base[case.offset:].view(case.rows, case.cols)
    if case.values == "wide":
        x.mul_(100)  # exp of an unshifted input overflows
    elif case.values == "special":
        x[torch.rand(x.shape, generator=gen, device=gen.device) < 0.01] = float("-inf")
        x[0] = float("-inf")
    elif case.values == "edge":  # for elementwise ops: values each op must pass through as torch does
        u = torch.rand(x.shape, generator=gen, device=gen.device)
        for i, v in enumerate((float("inf"), float("-inf"), float("nan"), 0.0, -0.0)):
            x[(u >= 0.01 * i) & (u < 0.01 * (i + 1))] = v
    return x


# How a reduction's input is laid out; each but "contiguous" takes a different path in the kernel
REDUCTION_LAYOUTS = (
    "contiguous",
    "offset",      # one element into its allocation, so not 16-byte aligned
    "padded",      # the last dim 3 elements apart more than its length
    "transposed",  # the last two dims swapped, so stride 1 along the second to last
    "strided",     # every other element along the last dim
)


@dataclass(frozen=True)
class ReductionCase:
    """The input's shape, dtype, layout and values, the dims reduced (an int, a tuple, or None
    for every dim), keepdim, and the correction that var and std take."""
    shape: tuple
    dim: int | tuple | None
    dtype: torch.dtype
    keepdim: bool = False
    layout: str = "contiguous"
    values: str = "normal"  # make_tensor's, or "ties" (small integers) or "shifted" (1000 + randn)
    seed: int = 0
    correction: int = 1

    def __str__(self):
        name = "x".join(map(str, self.shape)) or "scalar"
        name += f"-dim{self.dim}".replace(" ", "") + ("-keepdim" if self.keepdim else "")
        name += f"-{str(self.dtype).removeprefix('torch.')}-{self.layout}-{self.values}"
        return name + (f"-correction{self.correction}" if self.correction != 1 else "")


def reduction_dev_cases(bool_input: bool = False) -> list[ReductionCase]:
    """Reduced dims inner and outer, short to very long, ones that need the split path, length-1
    and empty dims, the layouts and the value kinds."""
    cases = []
    for d in DTYPES:
        cases += [ReductionCase((63, n), -1, d) for n in DEV_ROW_LENGTHS]
        cases += [ReductionCase((n, 63), 0, d) for n in (3, 17, 1021, 4099, 65536)]
        cases += [
            ReductionCase((63, 1021), -1, d, keepdim=True),
            ReductionCase((1021, 1021), 0, d),  # each reduced element of a column at a new alignment
            ReductionCase((7, 1021, 9), 1, d),
            ReductionCase((7, 1021, 9), (0, 2), d),  # two reduced dims that do not merge
            ReductionCase((5, 6, 7, 8), (1, 3), d, keepdim=True),
            ReductionCase((63, 1021), None, d),
            ReductionCase((2, 2**20 + 3), -1, d),  # a few long rows: split across blocks
            ReductionCase((2**20 + 3, 5), 0, d),
            ReductionCase((2**22,), None, d),
            ReductionCase((63, 1), -1, d),  # length-1 dims
            ReductionCase((1, 1021), 0, d),
            ReductionCase((5, 1, 7), 1, d),
            ReductionCase((), None, d),
            ReductionCase((63, 0), -1, d),  # empty reductions, which some ops allow
            ReductionCase((0, 63), 0, d),
            ReductionCase((0, 63), -1, d),  # no outputs
        ]
        for dim in (-1, 0):
            cases += [ReductionCase((63, 1021), dim, d, layout=layout) for layout in REDUCTION_LAYOUTS[1:]]
            cases += [ReductionCase((63, 1021), dim, d, values=v) for v in ("wide", "special", "edge", "ties", "shifted")]
        cases += [ReductionCase((63, 1021), -1, d, correction=c) for c in (0, 2)]
    if bool_input:
        cases += [ReductionCase((63, n), -1, torch.bool, values="edge") for n in (3, 17, 1021, 65536)]
        cases += [ReductionCase((1021, 63), 0, torch.bool, values="edge"), ReductionCase((63, 1021), None, torch.bool)]
    return cases


def reduction_heldout_cases(bool_input: bool = False, count: int = 100) -> list[ReductionCase]:
    """Cases from the same family as the dev cases, drawn from SOL_HELDOUT_SEED."""
    seed = os.environ.get("SOL_HELDOUT_SEED")
    if not seed:
        raise RuntimeError("set SOL_HELDOUT_SEED to run the held-out cases")
    rng = random.Random(seed + "-reduction")
    dtypes = DTYPES + ((torch.bool,) if bool_input else ())
    cases = []
    for _ in range(count):
        ndim = rng.randint(1, 4)
        shape = [max(1, round(2 ** rng.uniform(0, 22 / ndim))) if rng.random() < 0.9 else 1 for _ in range(ndim)]
        kind = rng.choice(("one", "one", "some", "all"))
        if kind == "one":
            dim = rng.randrange(-ndim, ndim)
        elif kind == "some":
            dim = tuple(sorted(rng.sample(range(ndim), rng.randint(1, ndim))))
        else:
            dim = None
        layouts = REDUCTION_LAYOUTS if ndim > 1 else ("contiguous", "offset", "padded", "strided")
        values = rng.choice(("normal", "normal", "wide", "special", "edge", "ties", "shifted"))
        cases.append(ReductionCase(tuple(shape), dim, rng.choice(dtypes), rng.random() < 0.3, rng.choice(layouts),
                                   values, seed=rng.getrandbits(63), correction=rng.choice((0, 1, 1))))
    return cases


def make_reduction_input(case: ReductionCase, gen: torch.Generator) -> torch.Tensor:
    """The case's input, in its layout. Callers seed gen with case.seed."""
    shape = list(case.shape)
    dtype = torch.float32 if case.dtype == torch.bool else case.dtype
    values = "normal" if case.values in ("ties", "shifted") else case.values
    alloc = list(shape)  # the shape that is allocated, before the view
    if case.layout == "padded":
        alloc[-1] += 3
    elif case.layout == "strided":
        alloc[-1] *= 2
    elif case.layout == "transposed":
        alloc[-2:] = alloc[:-3:-1]
    flat = Case(prod(alloc[:-1]), alloc[-1] if alloc else 1, dtype, offset=int(case.layout == "offset"), values=values)
    x = make_tensor(flat, gen).view(alloc)
    if case.layout == "padded":
        x = x[..., :shape[-1]]
    elif case.layout == "strided":
        x = x[..., ::2]
    elif case.layout == "transposed":
        x = x.transpose(-1, -2)
    if case.values == "ties":  # many equal values, so that argmax and max must break ties as torch does
        x.copy_((x * 2).round())
    elif case.values == "shifted":  # a mean far from 0: a variance taken as E[x^2] - E[x]^2 loses its digits
        x.add_(1000)
    return x != 0 if case.dtype == torch.bool else x
