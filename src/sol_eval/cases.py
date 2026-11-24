"""Correctness cases. Both softmax ops take a 2-D row-major tensor, so a case is a
shape, a dtype, an offset into the allocation, a kind of values and a seed for them."""

import os
import random
from dataclasses import dataclass

import torch

# float64 is left out: the kernels compute in float32 (see the softmax builders)
DTYPES = (torch.bfloat16, torch.float16, torch.float32)
DEV_ROW_LENGTHS = (3, 7, 8, 17, 64, 100, 257, 511, 633, 1021, 2049, 4099,
                   128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
EXTRA_ROW_LENGTHS = (7, 1021, 4099, 65536)


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int
    dtype: torch.dtype
    offset: int = 0         # elements between the start of the allocation and the tensor
    values: str = "normal"  # "normal", "wide" or "special"
    seed: int = 0

    def __str__(self):
        return f"{self.rows}x{self.cols}-{str(self.dtype).removeprefix('torch.')}-off{self.offset}-{self.values}"


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
    return x
