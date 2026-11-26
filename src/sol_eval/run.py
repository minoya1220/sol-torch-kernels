"""Check and time one op, or every op, on this machine's GPU.

    python -m sol_eval.run softmax_fwd [--heldout] [--json report.json]
    python -m sol_eval.run --all [--heldout] [--json report.json]

--heldout runs the held-out cases instead of the dev cases; it needs SOL_HELDOUT_SEED.
Exits with status 1 if any case fails.
"""

import argparse
import importlib.metadata
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

from sol_eval.check import check, check_graph
from sol_eval.measure import capture, measure, peak_bandwidth, time_graph
from sol_eval.ops import OPS


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
    g, copies = capture(fn, inputs, op.make_out(*inputs))
    us = time_graph(g)
    failure = check_graph(op, g, copies, make_inputs(op, replace(case, seed=case.seed + 1)))
    if failure:
        return {"case": str(case), "passed": False, "failure": failure}
    # torch is timed in its usual call, which allocates its own output: relu(x, out=) launches
    # relu and then a copy into out. The outputs given to measure only size the L2 rotation.
    torch_us = measure(lambda *xs, out: op.torch_call(*xs), inputs, op.make_out(*inputs))
    gbps = op.nbytes(*inputs) / us / 1e3
    return {"case": str(case), "passed": True, "failure": None, "us": us, "gbps": gbps,
            "sol": gbps / peak_gbps, "torch_us": torch_us, "speedup": torch_us / us,
            **time_compiled(op, inputs)}


def time_compiled(op, inputs) -> dict:
    """torch.compile of torch's usual call, timed like torch, for comparison only: it never
    fails a case. Compiled for these shapes and strides (dynamic=False) and as one graph
    (fullgraph=True), so a part dynamo cannot trace is an error rather than a silent fall
    back to eager. The first call compiles; its time is reported apart and not timed."""
    torch._dynamo.reset()  # each case compiles afresh, so no case runs into the recompile limit
    compiled = torch.compile(op.torch_call, dynamic=False, fullgraph=True)
    start = time.perf_counter()
    compiled_s = None
    try:
        compiled(*inputs)
        torch.cuda.synchronize()
        compiled_s = time.perf_counter() - start
        # an error here, in timing or capture, is reported like one in compiling
        compiled_us = measure(lambda *xs, out: compiled(*xs), inputs, op.make_out(*inputs))
    except Exception as e:
        return {"compiled_us": None, "compiled_s": compiled_s or time.perf_counter() - start,
                "compiled_error": f"{type(e).__name__}: {e}"}
    return {"compiled_us": compiled_us, "compiled_s": compiled_s, "compiled_error": None}


def correctness_cases(op, heldout):
    return op.heldout_cases() if heldout else op.dev_cases()


def run_op(op, heldout, peak) -> dict:
    compile_s = 0.0

    def build(*inputs):  # a kernel compiles when it is built, so the time spent here is compile time
        nonlocal compile_s
        start = time.perf_counter()
        fn = op.build(*inputs)
        compile_s += time.perf_counter() - start
        return fn

    start = time.perf_counter()
    correctness = run_correctness(op, build, correctness_cases(op, heldout))
    performance = [run_performance(op, build, case, peak) for case in op.perf_cases]
    torch_compile_s = sum(r.get("compiled_s", 0.0) for r in performance)
    run_s = time.perf_counter() - start - compile_s - torch_compile_s
    return {"op": op.name, "correctness": correctness, "performance": performance,
            "compile_s": compile_s, "torch_compile_s": torch_compile_s, "run_s": run_s}


def print_op(report):
    correctness = report["correctness"]
    passed = sum(r["passed"] for r in correctness)
    print(f"{report['op']}: {passed}/{len(correctness)} correctness cases pass")
    for r in correctness:
        if not r["passed"]:
            print(f"  FAIL {r['case']}: {r['failure']}")
    print(f"{'case':<48} {'us':>9} {'GB/s':>7} {'SoL':>6} {'torch us':>9} {'speedup':>8} {'compiled us':>12}")
    for r in report["performance"]:
        if r["passed"]:
            compiled = f"{r['compiled_us']:>12.1f}" if r["compiled_us"] is not None else f"{'error':>12}"
            print(f"{r['case']:<48} {r['us']:>9.1f} {r['gbps']:>7.0f} {r['sol']:>6.2f} {r['torch_us']:>9.1f} "
                  f"{r['speedup']:>8.2f} {compiled}")
        else:
            print(f"{r['case']:<48} FAIL {r['failure']}")
    for r in report["performance"]:
        if r.get("compiled_error"):
            print(f"  torch.compile failed on {r['case']}: {r['compiled_error']}")


def print_summary(reports):
    """One line per op, where the time went, then every failure. "vs compiled" is the
    speedup over torch.compile's call."""
    print(f"{'op':<28} {'correct':>9}  {'SoL':>11}  {'speedup':>11}  {'vs compiled':>11}")

    def span(values):
        return f"{min(values):.2f}-{max(values):.2f}" if values else "-"

    for report in reports:
        correctness, perf = report["correctness"], report["performance"]
        timed = [r for r in perf if r["passed"]]
        sol = span([r["sol"] for r in timed])
        speedup = span([r["speedup"] for r in timed])
        vs_compiled = span([r["compiled_us"] / r["us"] for r in timed if r["compiled_us"] is not None])
        passed = sum(r["passed"] for r in correctness)
        print(f"{report['op']:<28} {passed:>4}/{len(correctness):<4}  {sol:>11}  {speedup:>11}  {vs_compiled:>11}")
    compile_s, run_s = sum(r["compile_s"] for r in reports), sum(r["run_s"] for r in reports)
    torch_compile_s = sum(r["torch_compile_s"] for r in reports)
    print(f"{compile_s:.0f} s compiling, {torch_compile_s:.0f} s in torch.compile, {run_s:.0f} s checking and timing")
    for report in reports:
        for r in report["correctness"] + report["performance"]:
            if not r["passed"]:
                print(f"FAIL {report['op']} {r['case']}: {r['failure']}")
            elif r.get("compiled_error"):
                print(f"torch.compile failed on {report['op']} {r['case']}: {r['compiled_error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("op", choices=OPS, nargs="?")
    parser.add_argument("--all", action="store_true", help="every op, with one summary line each")
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.all == (args.op is not None):
        parser.error("give one op or --all")
    names = list(OPS) if args.all else [args.op]

    peak = peak_bandwidth()
    reports = [run_op(OPS[name], args.heldout, peak) for name in names]

    print(f"{torch.cuda.get_device_name()}, peak copy bandwidth {peak:.0f} GB/s")
    if args.all:
        print_summary(reports)
    else:
        print_op(reports[0])

    if args.json:
        report = {
            "heldout": args.heldout,
            "device": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
            "peak_gbps": peak,
            "ops": reports,
        }
        args.json.write_text(json.dumps(report, indent=2))

    all_passed = all(r["passed"] for report in reports for r in report["correctness"] + report["performance"])
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
