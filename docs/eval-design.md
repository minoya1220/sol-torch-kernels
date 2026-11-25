# Evaluation harness

The harness decides whether a kernel is correct and how fast it is. AI agents write the kernels, and the harness's verdict is their reward. That has two consequences. The owner must be able to read every line of the harness, so it stays small and plain. And it must hold up against an author who looks for gaps, so each gap that someone shows can be used gets a defence. `AGENTS.md` says how the repo is worked on.

The first version of this document (commit `c53bd7a`) was a full design: four isolated processes, static lint, audit hooks, CUPTI and `nsys` traces, sanitizers, a 30-flag catalogue and a 14-step plan. Most of that is kept below as a threat map rather than a build list. A defence from the map goes in when an exploit in `eval/exploits/` shows that the harness without it can be fooled.

## The kernel interface

An op is written as a builder, as in `minoya1220/kernels`:

```python
fn = softmax_fwd_builder(X)   # compiles for X's shape, dtype and pointer alignment
fn(X, out=Y)                  # launches on the current stream and returns Y
```

The builder compiles once; the returned function only launches. The harness builds once per case and then calls the function with tensors of the same shape, dtype and alignment. The torch baseline is written in the same form (`torch.ops.aten._softmax.out` and its backward), so the harness checks and times torch and the kernel with the same code.

Elementwise ops share one builder, `elementwise_builder(f, *inputs, out_like=, common_dtype=)`, where `f` is the op's function of float32 values. `sol_kernels/elementwise_ops.py` lists the ops as data: name, arity, `f` and the torch call to match. Run on meta tensors, the torch call gives the output's shape, strides and dtype (`out_like`), so the output is laid out as torch lays it out. The elementwise kernel is compiled per dtype, layout pattern and vector width, and takes shapes, strides and addresses at run time. It finds each row's alignment at run time and peels the unaligned head and tail of a row inside the same kernel.

The builder derives the vector width from the real pointer alignment of its tensors. The old builders passed `assumed_align=16` unconditionally, which is a false promise for a view at an odd offset.

## Correctness

### Reference

For each case the harness computes:

- `r`: the op in float64 on the case's input values, upcast (`torch.softmax(X.double())`, and the closed form `Y * (dY - sum(Y * dY))` for the backward);
- `t`: torch's own result at the case's dtype;
- `y`: the kernel's result.

### Error rule

Each op names its rule. Elementwise ops that are one correctly rounded float32 step in torch (add, sub, mul, div, sqrt, comparisons, where, clamp, relu, rounding and a few more; `exact=True` in the op list) must equal torch's result on every element, compared as values: NaN matches NaN, and +0 equals -0. There is one right answer, so any slack would only hide a bug. NaN payloads and the sign of zero are not compared because torch does not fix them: on the card, CUDA `maximum` and `minimum` returned a different zero on ties of +0 and -0 than torch on CPU does.

Every other op is measured against the float64 reference. Per element, with `eps` the machine epsilon of the output dtype:

```
s_i  = max(|r_i|, eps * max_j |r_j|, smallest normal of the dtype)    softmax forward
s_i  = max(|r_i|, (eps_f32 / eps) * Y_i * A, smallest normal)         softmax backward,
       with A = sum_j |Y_j dY_j| over the row
s_i  = max(|r_i|, smallest normal of the dtype)                       elementwise
e_i(x) = |x_i - r_i| / (eps * s_i)
pass: e_i(y) <= K * max(e_i(t), 1) for every i where r_i, rounded to the output dtype, is finite,
      with K = 4 for softmax and 2 for elementwise
```

Why this rule and not a fixed `atol` and `rtol`:

- The bound is relative to torch's own error on the same element, so it is tight where torch is accurate and loose only where the element is hard. There is no fixed number for an author to tune toward. Comparing maxima over the tensor instead let one hard element loosen the bound for all: torch's float32 gelu and silu lose every digit for very negative inputs, and a result 2^-10 off on every element passed.
- The floor of 1 (one ulp) lets a correctly rounded kernel pass where torch happens to be exact.
- For softmax forward, the scale `s_i` gives relative error for large entries and absolute error, at the size of the largest entry, for entries far below it. An elementwise output stands alone, so each element is measured against itself: under the softmax scale, zeroing every bf16 product below 1e-3 passed, and so would flushing subnormal results to zero.
- In the backward, dX_i = Y_i (dY_i - S) for the row sum S = sum_j Y_j dY_j. Where dY_i is close to S, dX_i is tiny and its error is Y_i times S's error. Torch's error there only says how close torch's sum landed on that row, so measured against |r_i| alone, a float32 sum as accurate as torch's but in another order failed 33 of 35 float32 cases on the card. A floating-point sum's error is a few eps times the sum of its terms' sizes, so the scale admits Y_i * A, in float32 units whatever the output dtype: S may carry a float32 sum's error and no more.
- The factor 4 allows a different summation order with float32 accumulation and rejects accumulation in a 16-bit type. `tests/test_check.py` checks both for the backward. An elementwise op has no sum to reorder, so 2 leaves room only for a different implementation of the same function.

Where `r_i`, rounded to the output dtype, is NaN or infinite, `y_i` must be NaN or the same infinity, so a result too large for the dtype must overflow. Measured as finite instead, such an element gave torch an infinite error and so an infinite bound. A NaN where the rounded `r_i` is finite makes `E` NaN, which fails.

### Cases

Dev cases are public and fixed. For both ops:

- 63 rows, each row length from the old repo's list (3 to 131072, powers of two and odd lengths), each of bf16, fp16 and fp32, contiguous, values from `randn`. float64 is left out because the kernels compute in float32;
- for a few row lengths and every dtype: the tensor at an offset of one element, which breaks 16-byte alignment; values scaled by 100, so an unshifted `exp` overflows; and special values, with some entries at `-inf` and one row entirely `-inf`.

Held-out cases come from the same public generator code, with random row counts, row lengths, dtypes, offsets and value kinds, drawn from a seed in the `SOL_HELDOUT_SEED` environment variable. The seed is never committed, since the repo is public. Fitting the generator's distribution is fine, because that is what correctness means. Learning the cases of one seed is not, which is why the seed stays out of the repo.

Elementwise ops get the softmax dev cases with every input contiguous, plus, on a 63 × 1021 shape and every dtype: each layout from `LAYOUTS` in `cases.py` (offset by one element, padded rows that each start at a new alignment, transposed, every other column, a row or column broadcast, a 0-dim float32 scalar) for one input at a time and for all inputs at once; some inputs of different dtypes; and edge values (`±inf`, `NaN`, `±0`). Held-out cases draw a layout and dtype for every input. Cases torch rejects, such as `lerp` with mixed dtypes, are dropped.

Rows of length 1 and non-contiguous rows are not covered for softmax, because the softmax builders do not support them (open question 3).

## Performance

Timing is `benchmark()` from `minoya1220/kernels`:

- Copy the call's tensors, keeping each input's offset and strides, until the copies total at least 1 GiB, or there are 100 copies, so the calls in one replay do not hit in L2. Very small cases still fit in L2 after 100 copies.
- Run 20 warmup calls on a side stream.
- Capture 100 calls, cycling through the copies, in one CUDA graph, which removes launch overhead from the number.
- Replay the graph 5 times, timed with CUDA events, and take the median replay time divided by 100.

Reported per perf case:

- time per call;
- bandwidth, from the bytes the op must move: 2 × the input size for the forward (read X, write Y), 3 × for the backward (read Y and dY, write dX);
- SoL: bandwidth divided by the peak, which is torch's `copy_` between two 1 GiB buffers, timed the same way on the same machine in the same run;
- speedup over torch's own op, timed the same way.

Elementwise ops are timed at 2^27 bf16 elements in three ways: contiguous, every input one element off alignment, and the second input a row broadcast. Softmax perf shapes are the old repo's `BENCH_DIMS` at 2^27 bf16 elements (256 MiB per tensor) rather than its 2^29, so that the float64 reference of the backward fits on a 16 GB card. Each perf case is also checked once, eagerly, before it is timed.

### Checking the timed graph

A correct eager call says nothing about what the captured graph does. After timing, the harness poisons every output copy with NaN, gives every input copy new values at the same addresses, replays the graph once more and checks every output copy against the new reference. This catches a kernel that skips work under capture and one that returns a cached result. Both cheats are in `eval/exploits/`.

## Report

`python -m sol_eval.run <op>` prints a table and writes JSON: device, torch and CuTe DSL versions, peak, then each correctness case (pass or fail, with the error) and each perf case (time, GB/s, SoL, torch time, speedup). `python -m sol_eval.run --all` does the same for every op in one process, with one peak measurement, and prints one line per op and then every failure.

Compiling dominates a run. The elementwise kernel takes shapes and strides at run time, so one compile serves every case with the same dtypes, layout pattern and vector width, and `compile_kernel` keeps each compile in an `lru_cache`. Everything runs in one process, so a compile never runs while a kernel is timed. Softmax still compiles once per case, for its row length.

## Threat map

**This round** means the current stack of PRs defends it, with an exploit where the defence is more than a one-line rule. **Not yet** means no exploit has shown that the harness is fooled. Letters refer to the attack catalogue in commit `c53bd7a`, section 2.1, which has the details.

| Attack | Defence | Status |
|---|---|---|
| Torch, cuBLAS or Triton does the work inside the launcher (A1, A4) | Every kernel in a checked window must be one the impl compiled: kernel-name trace through `torch.profiler` | not yet (owner's call: too strict for now) |
| Load a precompiled cubin under an impl kernel name (A2) | Record the modules the impl compiled; ban `ctypes` and `cuda.bindings` in impl files | not yet |
| Use memcpy or memset instead of a kernel (A3) | The same trace | not yet |
| Branch on the visible dev shapes or values (B1, B2) | Held-out cases from a seed kept out of the repo | this round |
| Correct only for the value ranges random tests hit (B3) | Special values in dev cases; every bit pattern for 16-bit unary ops | partly: special values only |
| Behave differently under graph capture (B4, D4, E1) | Poison the outputs, replay the timed graph, check every copy | this round |
| Cache results by pointer, or compute the answer at build time (D1, E4) | New input values at the same addresses before the check replay | this round |
| Sit just inside a fixed tolerance, or compute in a lower precision (C1, C2, C3) | Bound relative to torch's own error per case | this round |
| Bias every element while staying under the max bound (C4) | Mean error rule | not yet |
| NaN or inf handled wrongly (C5, J2) | Class check before the error rule | this round |
| Keep state on the device so that only the first replay is right (D3) | The check replay runs after the timed replays | this round, no exploit yet |
| Pipeline across calls so that the last call is unfinished (E7) | The check replay: the last output is still poison | this round |
| Run work outside the timed window: side streams, persistent kernels, host threads (E2, E3) | Trace spans, device-wide syncs, thread count | not yet |
| Patch the timer or the checker from an impl import (E6) | Tripwire on harness functions; external `nsys` timing | not yet |
| Claim more bytes than the op moves (F1) | Bytes come from the harness, never from the impl | this round |
| Time with data in L2 (F2) | 1 GiB of copies per replay; reject SoL above 1.02 | partly: copies only |
| Slow the peak or the baseline (F3) | Measure them in a process that never loads impl code | not yet |
| Tune only for the bench shapes, or keep stale tunings (G1 to G4) | Held-out perf shapes; kernel hash in tuning rows | not yet: no tuner |
| Edit the harness, tests, CI or lockfile in a kernel PR (H1 to H3) | The owner reviews and merges every PR (`AGENTS.md` rule 8) | process |
| Read the held-out seed (H4) | The seed lives only in the environment of the evaluation run | this round |
| Out-of-bounds access, races (I1 to I3) | Red zones; `compute-sanitizer` | not yet |
| Assume an alignment the input does not have (I4) | Cases at an offset of one element | this round |
| 32-bit index overflow (I5) | Cases over 2^31 elements | not yet |
| Nondeterminism (J1) | Run twice and compare bitwise | not yet |
| Non-contiguous, broadcast and aliased layouts (J3, J4) | Layout cases | not yet: softmax takes contiguous rows |

Also deferred: a separate verifier process that holds the seed and judges outputs it did not produce, NVML clock and power monitoring, and CI on a GPU.

## Open questions

1. **Acceptance gate.** What makes a kernel PR mergeable? A proposal: every dev and held-out case passes, and no perf case is slower than torch. A SoL target per op can come once the first ops show what is reachable.
2. **Reference GPU.** The RTX 5060 Ti and 5090 differ about 4× in bandwidth and 3× in L2. Report per device and gate on one of them?
3. **Coverage.** Should the softmax kernels support rows of length 1 and non-contiguous rows, so that the harness tests every shape torch accepts?
