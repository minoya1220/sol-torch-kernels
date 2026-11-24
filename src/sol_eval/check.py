"""Correctness: a result passes if it matches torch bit for bit (for ops that are one
correctly rounded step), or if, on every element, its error against the float64 reference
is at most K times torch's own error on that element. docs/eval-design.md says why."""

import torch

# K per error rule. "max_scaled" (softmax) measures small entries against the largest one;
# "relative" (elementwise ops) measures every entry against itself.
K = {"max_scaled": 4, "relative": 2}


def error(x: torch.Tensor, ref: torch.Tensor, rule: str) -> torch.Tensor:
    """Error of each element of x, in units of eps of x's dtype, over the elements where ref
    rounded to x's dtype is finite."""
    info = torch.finfo(x.dtype)
    finite = ref.to(x.dtype).isfinite()
    r = ref[finite]
    floor = info.tiny
    if rule == "max_scaled" and r.numel() > 0:
        floor = max(info.eps * r.abs().max().item(), info.tiny)
    scale = r.abs().clamp_min(floor)
    return (x.double()[finite] - r).abs() / (info.eps * scale)


def bits(x: torch.Tensor) -> torch.Tensor:
    return x.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()])


def compare(y: torch.Tensor, t: torch.Tensor, ref: torch.Tensor, rule: str) -> str | None:
    """None if y passes under the op's rule, else why it fails. t is torch's result on the
    same inputs."""
    if y.shape != t.shape or y.dtype != t.dtype:
        return f"got {tuple(y.shape)} {y.dtype}, torch gives {tuple(t.shape)} {t.dtype}"
    if rule == "exact":
        # NaN payloads are not compared: torch and the kernel may build NaN differently
        same = bits(y) == bits(t)
        if y.is_floating_point():
            same |= y.isnan() & t.isnan()
        wrong = (~same).sum().item()
        return f"{wrong} of {y.numel()} elements differ from torch" if wrong else None
    # A reference too large for y's dtype rounds to inf, and y must be inf there too.
    # Measured as finite, it gave torch an infinite error and so an infinite bound.
    rounded = ref.to(y.dtype)
    special = ~rounded.isfinite()
    ys, rs = y[special].double(), rounded[special].double()
    nan = rs.isnan()
    if not torch.equal(ys.isnan(), nan) or not torch.equal(ys[~nan], rs[~nan]):
        return "does not match the reference where it is NaN or inf"
    # Each element against torch's error on the same element: one element where torch's
    # formula underflows must not loosen the bound on all the others.
    e_y, e_t = error(y, ref, rule), error(t, ref, rule)
    bound = K[rule] * e_t.clamp_min(1.0)
    bad = ~(e_y <= bound)  # also catches NaN
    if bad.any():
        i = (e_y / bound).nan_to_num(float("inf")).argmax()
        return (f"{bad.sum().item()} elements over the bound; worst error {e_y[i].item():.3g} ulp, "
                f"bound {bound[i].item():.3g} (torch {e_t[i].item():.3g})")
    return None


def references(op, inputs) -> tuple[torch.Tensor, torch.Tensor]:
    """torch's result and the float64 reference for these inputs."""
    t = op.make_out(*inputs)
    op.torch_fn(*inputs, out=t)
    return t, op.ref64(*inputs)


def check(op, fn, inputs) -> str | None:
    """Run fn once on inputs and compare its output; None if it passes."""
    t, ref = references(op, inputs)
    y = op.make_out(*inputs)
    fn(*inputs, out=y)
    return compare(y, t, ref, op.rule)


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
        failure = compare(y, t, ref, op.rule)
        if failure:
            return f"after a graph replay: {failure}"
    return None
