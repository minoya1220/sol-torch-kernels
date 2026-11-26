"""CUDA-graph timing, ported from benchmark() in minoya1220/kernels utils/measure.py.
The harness always calls fn(*inputs, out=out), so the general argument handling is gone."""

from statistics import median

import torch

L2_BUFFER_SIZE = 2**30  # 1 GiB is larger than L2 on any current GPU


def copy_view(t):
    """A copy of t's whole allocation, viewed the same way, so that the copy keeps t's
    offset, strides and so its alignment. clone() would make it contiguous."""
    storage = t.untyped_storage().clone()
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(storage, t.storage_offset(), t.shape, t.stride())


def make_copies(inputs, out, iters):
    """Copies of the call's tensors, enough that one replay of `iters` calls touches at least
    1 GiB, so that no call finds its data in L2."""
    size = sum(t.untyped_storage().nbytes() for t in inputs) + out.nbytes
    n = min(iters, -(-L2_BUFFER_SIZE // size))
    return [(tuple(copy_view(t) for t in inputs), out.clone()) for _ in range(n)]


ITERS = 100  # calls per graph replay


def capture(fn, inputs, out, *, warmup=20):
    """Warm fn up, then capture ITERS calls of it in one CUDA graph, cycling through copies of
    the call's tensors. Returns the graph and the copies it reads and writes."""
    copies = make_copies(inputs, out, ITERS)

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
        for i in range(ITERS):
            x, y = copies[i % len(copies)]
            fn(*x, out=y)
    g.replay()
    torch.cuda.synchronize()
    return g, copies


def time_graph(g, *, repeats=5) -> float:
    """Median time of one call in the graph, in microseconds."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        start.record()
        g.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000 / ITERS)
    return median(times)


def measure(fn, inputs, out) -> float:
    """Median time of one call of fn(*inputs, out=out), in microseconds."""
    g, _ = capture(fn, inputs, out)
    return time_graph(g)


def peak_bandwidth() -> float:
    """GB/s of torch's copy_ between two 1 GiB buffers, timed the same way as a kernel."""
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    us = measure(lambda x, *, out: out.copy_(x), (src,), torch.empty_like(src))
    return 2 * src.nbytes / us / 1e3


def peak_matmul() -> float:
    """TFLOP/s of torch's bf16 matmul of two 8192 x 8192 matrices, timed the same way as a
    kernel. cuBLAS accumulates in float32, as the attention kernel does: some cards run a
    16-bit accumulator faster, and the kernel cannot use that rate."""
    n = 8192
    a, b = (torch.randn(n, n, device="cuda").bfloat16() for _ in range(2))
    us = measure(lambda a, b, *, out: torch.matmul(a, b, out=out), (a, b), torch.empty_like(a))
    return 2 * n**3 / us / 1e6
