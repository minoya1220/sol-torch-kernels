"""Correctness cases. Both softmax ops take a 2-D row-major tensor, so a case is a
shape, a dtype, an offset into the allocation, a kind of values and a seed for them.
An elementwise op's inputs broadcast to the case's shape, and each can have its own layout
and dtype."""

import os
import random
from dataclasses import dataclass, replace

import torch

# float64 is left out: the softmax kernels compute in float32 (see the softmax builders)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
# dtypes that only elementwise ops take
EXTRA_DTYPES = (torch.float64, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)
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
    "scalar",      # a 0-dim tensor: float32 for a float case, int64 for an integer one, else bool
)
# input dtypes for cases that mix them: bf16 with fp16 computes in fp32, uint8 with int8 in
# int16, and an integer or bool with a float in the float
MIXED_DTYPES = ((torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16),
                (torch.bfloat16, torch.float32), (torch.float32, torch.float16),
                (torch.float64, torch.float16), (torch.uint8, torch.int8), (torch.int32, torch.bfloat16),
                (torch.int64, torch.float32), (torch.bool, torch.float16))


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
    """The softmax dev cases, with every input contiguous, and the same on fewer row lengths
    for the dtypes only elementwise ops take; then, for every dtype, each layout and edge
    values on one odd shape; then some mixed dtypes."""
    cases = dev_cases()
    for d in EXTRA_DTYPES:
        cases += [Case(63, n, d) for n in EXTRA_ROW_LENGTHS]
        cases += [Case(63, 1021, d, offset=1), Case(63, 1021, d, values="wide")]
    for d in DTYPES + EXTRA_DTYPES:
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
    """heldout_cases, with a layout and a dtype drawn for every input. Four in ten cases
    take a dtype only elementwise ops take, and each input keeps the case's dtype two
    times in three."""
    rng = random.Random(os.environ.get("SOL_HELDOUT_SEED", "") + f"-elementwise-{arity}")
    cases = []
    for case in heldout_cases(count):
        dtype = case.dtype if rng.random() < 0.6 else rng.choice(EXTRA_DTYPES)
        layouts = tuple(rng.choice(LAYOUTS[:1] * 3 + LAYOUTS[1:]) for _ in range(arity))
        dtypes = tuple(dtype if rng.random() < 2 / 3 else rng.choice(DTYPES + EXTRA_DTYPES) for _ in range(arity))
        values = rng.choice((case.values, "edge"))
        cases.append(replace(case, dtype=dtype, values=values, layouts=layouts, dtypes=dtypes))
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
        if c.dtype.is_floating_point:
            scalar_dtype = torch.float32
        else:
            scalar_dtype = torch.bool if c.dtype == torch.bool else torch.int64
        return make_tensor(replace(c, rows=1, cols=1, dtype=scalar_dtype), gen)[0, 0]
    raise ValueError(layout)


def make_tensor(case: Case, gen: torch.Generator) -> torch.Tensor:
    """A tensor for the case, on the generator's device. Callers seed gen with case.seed."""
    n = case.rows * case.cols
    if case.dtype == torch.bool:
        base = torch.randn(case.offset + n, generator=gen, device=gen.device) > 0
        return base[case.offset:].view(case.rows, case.cols)
    if not case.dtype.is_floating_point:
        return make_integers(case, gen)[case.offset:].view(case.rows, case.cols)
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


def make_integers(case: Case, gen: torch.Generator) -> torch.Tensor:
    """The case's allocation of integers: for "wide" values, any value of the dtype; else
    values from -70 to 70 (wrapped into uint8), which as shift amounts are in range and out
    of it. "edge" and "special" mix in the dtype's limits, 0, 1 and -1."""
    size = case.offset + case.rows * case.cols
    info = torch.iinfo(case.dtype)
    if case.values == "wide":
        return torch.randint(info.min, info.max, (size,), generator=gen, device=gen.device, dtype=case.dtype)
    x = torch.randint(-70, 71, (size,), generator=gen, device=gen.device).to(case.dtype)
    if case.values in ("edge", "special"):
        u = torch.rand(size, generator=gen, device=gen.device)
        edges = [v for v in (info.min, info.max, 0, 1, -1) if v >= info.min]
        for i, v in enumerate(edges):
            x[(u >= 0.01 * i) & (u < 0.01 * (i + 1))] = v
    return x
