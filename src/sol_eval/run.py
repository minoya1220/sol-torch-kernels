"""Check and time one op on this machine's GPU.

    python -m sol_eval.run softmax_fwd [--heldout] [--json report.json]

--heldout runs the held-out cases instead of the dev cases; it needs SOL_HELDOUT_SEED.
Exits with status 1 if any case fails.
"""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

from sol_eval.cases import Case, dev_cases, heldout_cases
from sol_eval.check import check
from sol_eval.measure import measure, peak_bandwidth
from sol_eval.ops import OPS

# 2^27 bf16 elements is 256 MiB per tensor, far above L2, while the float64
# reference for the backward still fits on a 16 GB card
PERF_ELEMENTS = 2**27
PERF_ROW_LENGTHS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)


def make_inputs(op, case):
    return op.make_inputs(case, torch.Generator("cuda").manual_seed(case.seed))


def run_correctness(op, build, cases) -> list[dict]:
    results = []
    for case in cases:
        inputs = make_inputs(op, case)
        try:
            failure = check(op, build(*inputs), inputs)
        except Exception as e:  # a crash fails the case
            failure = f"{type(e).__name__}: {e}"
        results.append({"case": str(case), "passed": failure is None, "failure": failure})
    return results


def run_performance(op, build, case, peak_gbps) -> dict:
    inputs = make_inputs(op, case)
    fn = build(*inputs)
    failure = check(op, fn, inputs)
    if failure:
        return {"case": str(case), "passed": False, "failure": failure}
    us = measure(fn, inputs, torch.empty_like(inputs[0]))
    torch_us = measure(op.torch_fn, inputs, torch.empty_like(inputs[0]))
    gbps = op.nbytes(*inputs) / us / 1e3
    return {"case": str(case), "passed": True, "failure": None, "us": us, "gbps": gbps,
            "sol": gbps / peak_gbps, "torch_us": torch_us, "speedup": torch_us / us}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("op", choices=OPS)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    op = OPS[args.op]

    cases = heldout_cases() if args.heldout else dev_cases()
    correctness = run_correctness(op, op.build, cases)
    peak = peak_bandwidth()
    perf_cases = [Case(PERF_ELEMENTS // n, n, torch.bfloat16) for n in PERF_ROW_LENGTHS]
    performance = [run_performance(op, op.build, case, peak) for case in perf_cases]

    passed = sum(r["passed"] for r in correctness)
    print(f"{op.name} on {torch.cuda.get_device_name()}: {passed}/{len(correctness)} correctness cases pass")
    for r in correctness:
        if not r["passed"]:
            print(f"  FAIL {r['case']}: {r['failure']}")
    print(f"peak copy bandwidth {peak:.0f} GB/s")
    print(f"{'case':<32} {'us':>9} {'GB/s':>7} {'SoL':>6} {'torch us':>9} {'speedup':>8}")
    for r in performance:
        if r["passed"]:
            print(f"{r['case']:<32} {r['us']:>9.1f} {r['gbps']:>7.0f} {r['sol']:>6.2f} {r['torch_us']:>9.1f} {r['speedup']:>8.2f}")
        else:
            print(f"{r['case']:<32} FAIL {r['failure']}")

    if args.json:
        report = {
            "op": op.name,
            "heldout": args.heldout,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "peak_gbps": peak,
            "correctness": correctness,
            "performance": performance,
        }
        args.json.write_text(json.dumps(report, indent=2))

    all_passed = all(r["passed"] for r in correctness + performance)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
