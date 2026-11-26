"""CUDA-graph timing, ported from benchmark() in minoya1220/kernels utils/measure.py.
The harness always calls fn(*inputs, out=out), so the general argument handling is gone."""

import threading
from statistics import median

import pynvml
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


class Clocks:
    """Samples the SM and memory clocks through NVML on a thread while the block runs, as
    MeasureMemoryClock does in minoya1220/kernels utils/measure.py, and keeps the medians in
    MHz. A card's clock moves with its power and temperature, so a rate is only comparable
    with a ceiling at the clock it ran at. Reading clocks needs no root; locking them with
    nvidia-smi -lgc does, and when they are locked the samples show it. NVML gives each clock
    averaged over its own sample period, 1/6 s to 1 s by product, so a timing should last a
    second or more."""

    def __init__(self, interval=0.01):
        self.interval = interval

    def sample(self):
        while not self.stop.is_set():
            self.samples.append((pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM),
                                 pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)))
            self.stop.wait(self.interval)

    def __enter__(self):
        pynvml.nvmlInit()
        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}")
        self.samples, self.stop = [], threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sm_mhz = median(sm for sm, _ in self.samples)
        self.mem_mhz = median(mem for _, mem in self.samples)


def time_graph(g, *, repeats=5) -> tuple[float, Clocks]:
    """Median time of one call in the graph, in microseconds, and the clocks it ran at."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    with Clocks() as clocks:
        for _ in range(repeats):
            start.record()
            g.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000 / ITERS)
    return median(times), clocks


def measure(fn, inputs, out) -> tuple[float, Clocks]:
    """Median time of one call of fn(*inputs, out=out), in microseconds, and its clocks."""
    g, _ = capture(fn, inputs, out)
    return time_graph(g)


def peak_bandwidth() -> tuple[float, Clocks]:
    """GB/s of torch's copy_ between two 1 GiB buffers, timed the same way as a kernel."""
    src = torch.empty(2**30, dtype=torch.uint8, device="cuda")
    us, clocks = measure(lambda x, *, out: out.copy_(x), (src,), torch.empty_like(src))
    return 2 * src.nbytes / us / 1e3, clocks


def peak_flops_per_sm_clock() -> tuple[float, Clocks]:
    """Tensor-core FLOPs per SM per SM clock of the bf16 mma.sync loop in sol_eval.mma_peak,
    timed the same way as a kernel at the clock it ran at."""
    from sol_eval.mma_peak import mma_loop
    inputs, out, flops, fn = mma_loop()
    us, clocks = measure(fn, inputs, out)
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return flops / (us * 1e-6) / (sms * clocks.sm_mhz * 1e6), clocks


def ceiling_tflops(flops_per_sm_clock: float, sm_mhz: float) -> float:
    """TFLOP/s the tensor cores can do at this SM clock."""
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return sms * flops_per_sm_clock * sm_mhz * 1e6 / 1e12
