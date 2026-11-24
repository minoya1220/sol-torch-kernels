"""CUDA-graph timing, ported from benchmark() in minoya1220/kernels utils/measure.py.
The harness always calls fn(*inputs, out=out), so the general argument handling is gone."""

from statistics import median

import torch

L2_BUFFER_SIZE = 2**30  # 1 GiB is larger than L2 on any current GPU


def make_copies(inputs, out, iters):
    """Copies of the call's tensors, enough that one replay of `iters` calls touches at least
    1 GiB, so that no call finds its data in L2."""
    size = sum(t.nbytes for t in inputs) + out.nbytes
    n = min(iters, -(-L2_BUFFER_SIZE // size))
    return [(tuple(t.clone() for t in inputs), out.clone()) for _ in range(n)]


def measure(fn, inputs, out, *, warmup=20, iters=100, repeats=5) -> float:
    """Median time of one call of fn(*inputs, out=out), in microseconds."""
    copies = make_copies(inputs, out, iters)

    # launching on a side stream is only needed if fn caches the stream it was first called on
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(warmup):
            x, y = copies[i % len(copies)]
            fn(*x, out=y)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            x, y = copies[i % len(copies)]
            fn(*x, out=y)
    g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / iters)
    return median(times)


def peak_bandwidth() -> float:
    """GB/s of torch's copy_ between two 1 GiB buffers, timed the same way as a kernel."""
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    us = measure(lambda x, *, out: out.copy_(x), (src,), torch.empty_like(src))
    return 2 * src.nbytes / us / 1e3
