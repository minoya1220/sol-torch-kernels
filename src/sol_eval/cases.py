"""Correctness cases. Both softmax ops take a 2-D row-major tensor, so a case is a
shape, a dtype, an offset into the allocation, a kind of values and a seed for them.
An elementwise op's inputs broadcast to the case's shape, and each can have its own layout
and dtype. An attention case gives the shapes of q, k and v, a dtype, the causal flag and the
scale, a kind of values and a seed."""

import os
import random
from dataclasses import dataclass, replace

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


@dataclass(frozen=True)
class AttentionCase:
    """q is (batch, heads, sq, head_dim); k and v are (batch, kv_heads, sk, head_dim)."""
    batch: int
    heads: int
    kv_heads: int
    sq: int
    sk: int
    head_dim: int
    dtype: torch.dtype
    causal: bool = False
    scale: float | None = None  # None is torch's default, 1 / sqrt(head_dim)
    values: str = "normal"      # "normal", or "wide": q times 30, so that exp of an unshifted score overflows
    seed: int = 0

    def __str__(self):
        heads = f"{self.heads}/{self.kv_heads}" if self.kv_heads != self.heads else f"{self.heads}"
        return (f"{self.batch}x{heads}x{self.sq}x{self.sk}x{self.head_dim}-{str(self.dtype).removeprefix('torch.')}"
                f"-{'causal' if self.causal else 'full'}-scale{self.scale or 'default'}-{self.values}")


# the kernels take 16-bit inputs and accumulate in float32
ATTENTION_DTYPES = (torch.bfloat16, torch.float16)
# (sq, sk): a multiple of every tile, lengths off the tiles, sq != sk both ways, length 1,
# and a long row, where rounding error grows with the number of keys
ATTENTION_LENGTHS = ((128, 128), (257, 257), (1, 1), (1, 1000), (1000, 1), (100, 333), (333, 100), (64, 8192))


def attention_dev_cases() -> list[AttentionCase]:
    cases = []
    for d in (64, 128):
        for dtype in ATTENTION_DTYPES:
            for causal in (False, True):
                for sq, sk in ATTENTION_LENGTHS:
                    cases.append(AttentionCase(2, 4, 4, sq, sk, d, dtype, causal))
                cases.append(AttentionCase(2, 8, 2, 257, 257, d, dtype, causal))  # grouped-query
                cases.append(AttentionCase(2, 4, 4, 257, 257, d, dtype, causal, scale=0.3))
                cases.append(AttentionCase(2, 4, 4, 257, 257, d, dtype, causal, values="wide"))
    return cases


def attention_heldout_cases(count: int = 100) -> list[AttentionCase]:
    """attention_dev_cases' kinds of case, with random lengths, from the held-out seed."""
    seed = os.environ.get("SOL_HELDOUT_SEED")
    if not seed:
        raise RuntimeError("set SOL_HELDOUT_SEED to run the held-out cases")
    rng = random.Random(seed + "-attention")
    cases = []
    for _ in range(count):
        kv_heads = rng.choice((1, 2, 4))
        heads = kv_heads * rng.choice((1, 1, 2, 4))
        sq, sk = (max(1, round(2 ** rng.uniform(0, 12))) for _ in range(2))
        scale = rng.choice((None, None, rng.uniform(0.02, 1.0)))
        cases.append(AttentionCase(rng.randint(1, 3), heads, kv_heads, sq, sk, rng.choice((64, 128)),
                                   rng.choice(ATTENTION_DTYPES), rng.choice((False, True)), scale,
                                   rng.choice(("normal", "normal", "wide")), seed=rng.getrandbits(63)))
    return cases


def make_attention_inputs(case: AttentionCase, gen: torch.Generator) -> tuple[torch.Tensor, ...]:
    """q, k and v for the case, on the generator's device. Callers seed gen with case.seed."""
    def randn(heads, seq):
        shape = (case.batch, heads, seq, case.head_dim)
        return torch.randn(shape, generator=gen, device=gen.device).to(case.dtype)

    q, k, v = randn(case.heads, case.sq), randn(case.kv_heads, case.sk), randn(case.kv_heads, case.sk)
    if case.values == "wide":
        q.mul_(30)
    return q, k, v
