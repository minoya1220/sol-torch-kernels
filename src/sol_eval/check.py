"""Correctness: a result passes if its error against the float64 reference is at most
K times torch's own error on the same inputs. docs/eval-design.md says why."""

import torch

K = 4


def error(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Largest error of x, in units of eps of x's dtype, over the elements where ref rounded
    to x's dtype is finite. Small entries are measured against the largest entry, not
    against themselves."""
    info = torch.finfo(x.dtype)
    finite = ref.to(x.dtype).isfinite()
    r = ref[finite]
    if r.numel() == 0:
        return 0.0
    scale = r.abs().clamp_min(max(info.eps * r.abs().max().item(), info.tiny))
    return ((x.double()[finite] - r).abs() / (info.eps * scale)).max().item()


def compare(y: torch.Tensor, t: torch.Tensor, ref: torch.Tensor) -> str | None:
    """None if y passes, else why it fails. t is torch's result on the same inputs."""
    if y.shape != t.shape or y.dtype != t.dtype:
        return f"got {tuple(y.shape)} {y.dtype}, torch gives {tuple(t.shape)} {t.dtype}"
    if y.dtype == torch.bool:  # comparisons have exactly one right answer
        wrong = (y != ref).sum().item()
        return f"{wrong} of {y.numel()} elements differ from the reference" if wrong else None
    # A reference too large for y's dtype rounds to inf, and y must be inf there too.
    # Measured as finite, it gave torch an infinite error and so an infinite bound.
    rounded = ref.to(y.dtype)
    special = ~rounded.isfinite()
    ys, rs = y[special].double(), rounded[special].double()
    nan = rs.isnan()
    if not torch.equal(ys.isnan(), nan) or not torch.equal(ys[~nan], rs[~nan]):
        return "does not match the reference where it is NaN or inf"
    e_y, e_t = error(y, ref), error(t, ref)
    bound = K * max(e_t, 1.0)
    if not e_y <= bound:  # also fails when e_y is NaN
        return f"error {e_y:.3g} ulp, bound {bound:.3g} (torch {e_t:.3g})"
    return None


def references(op, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """torch's result and the float64 reference for these inputs."""
    t = torch.empty_like(inputs[0])
    op.torch_fn(*inputs, out=t)
    return t, op.ref64(*inputs)


def check(op, fn, inputs) -> str | None:
    """Run fn once on inputs and compare its output; None if it passes."""
    t, ref = references(op, inputs)
    y = torch.empty_like(inputs[0])
    fn(*inputs, out=y)
    return compare(y, t, ref)


def check_graph(op, g, copies, new_inputs) -> str | None:
    """Give every input copy new values at the same addresses, poison every output copy with
    NaN, replay the timed graph once, and compare every output copy. A call that did its work
    eagerly but not under the graph leaves NaN behind; one that returns a result cached from
    earlier inputs returns the wrong values."""
    for x, y in copies:
        for a, b in zip(x, new_inputs):
            a.copy_(b)
        y.fill_(float("nan"))
    g.replay()
    t, ref = references(op, new_inputs)
    for _, y in copies:
        failure = compare(y, t, ref)
        if failure:
            return f"after a graph replay: {failure}"
    return None
