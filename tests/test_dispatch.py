"""CPU tests for make_launcher, BoundLauncher's spec check, the launcher cache and the public path (design 6.3, 6.5).

The fake impl here compiles nothing; its launcher records what it was given. The
conversion to cute tensors (`from_dlpack`) is real and works on CPU tensors.
"""

from __future__ import annotations

import pytest
import torch

from sol_kernels._runtime import bridge, dispatch
from sol_kernels._runtime.interface import CallSpec, Impl, NotSupported, Param, Registry, TensorSpec

STREAM = object()


class Fake(Impl):
    op = "fake"
    search_space = {"bdim": Param((128, 256), 256)}
    supported = True
    ws_bytes = 0
    ret = None
    use_ctx_later = False

    def __init__(self):
        self.builds = 0
        self.calls = []
        self.ctx = None

    def supports(self, spec):
        return self.supported

    def valid_config(self, spec, cfg):
        return cfg["bdim"] != 128 or spec.inputs[0].numel < 10**6

    def workspace_bytes(self, spec, cfg):
        return self.ws_bytes

    def build(self, spec, cfg, ctx):
        self.builds += 1
        self.ctx = ctx
        ins, outs, ws = ctx.dummies()
        assert len(ins) == len(spec.inputs) and len(outs) == len(spec.outputs)
        assert (ws is None) == (self.ws_bytes == 0)
        calls, ret = self.calls, self.ret

        def launch(inputs, outputs, workspace, scalars, stream):
            calls.append(([tuple(t.shape) for t in inputs], [tuple(t.shape) for t in outputs], workspace, scalars, stream))
            return ret

        return launch


def aligned(n=4096, dtype=torch.float32):
    return bridge.alloc_with_layout(TensorSpec((n,), (1,), bridge.DTYPE_NAMES[dtype], 0, 256), "cpu", poison=False)


def tensors():
    b = aligned()
    return b[:32].view(4, 8), b[1024:1056].view(4, 8)


def spec_of(x, y, scalars=()):
    return bridge.call_spec("fake", [x], [y], scalars=scalars)


def test_build_and_call():
    x, y = tensors()
    impl = Fake()
    L = dispatch.make_launcher(impl, spec_of(x, y))
    assert impl.builds == 1 and L.cfg == {"bdim": 256}
    L([x], [y], None, (), STREAM)
    assert impl.calls == [([(4, 8)], [(4, 8)], None, (), STREAM)]


def test_context_is_closed_after_build():
    x, y = tensors()
    impl = Fake()
    dispatch.make_launcher(impl, spec_of(x, y))
    for use in (impl.ctx.dummies, impl.ctx.stream, lambda: impl.ctx.compile(lambda: None)):
        with pytest.raises(dispatch.ImplContractError):
            use()


def test_unsupported_spec_never_falls_back():
    x, y = tensors()
    impl = Fake()
    impl.supported = False
    with pytest.raises(NotSupported):
        dispatch.make_launcher(impl, spec_of(x, y))
    assert impl.builds == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b, x, y: ([b[2048:2080].view(8, 4)], [y]),           # shape
        lambda b, x, y: ([b[2049:2081].view(4, 8)], [y]),           # alignment (offset 1 element)
        lambda b, x, y: ([b[2048:2112:2].view(4, 8)], [y]),         # strides
        lambda b, x, y: ([b[2048:2080].view(4, 8).double()], [y]),  # dtype
        lambda b, x, y: ([x], [x]),                                 # now in place
        lambda b, x, y: ([b[2112:2144].view(4, 8)], [y]),           # storage offset, same alignment
    ],
    ids=["shape", "align", "stride", "dtype", "alias", "offset"],
)
def test_call_with_a_different_spec_is_refused(mutate):
    b = aligned()
    x, y = b[:32].view(4, 8), b[1024:1056].view(4, 8)
    impl = Fake()
    L = dispatch.make_launcher(impl, spec_of(x, y))
    ins, outs = mutate(b, x, y)
    with pytest.raises(dispatch.SpecMismatch):
        L(ins, outs, None, (), STREAM)
    assert impl.calls == []


def test_scalar_kinds_are_checked():
    x, y = tensors()
    L = dispatch.make_launcher(Fake(), spec_of(x, y, scalars=(1.0,)))
    L([x], [y], None, (2.5,), STREAM)
    with pytest.raises(dispatch.SpecMismatch):
        L([x], [y], None, (2,), STREAM)
    with pytest.raises(dispatch.SpecMismatch):
        L([x], [y], None, (), STREAM)


@pytest.mark.parametrize("cfg", [{"bdim": 64}, {"bdim": True}, {}, {"bdim": 256, "extra": 1}, {"bdim": 256.0}])
def test_bad_configs_rejected(cfg):
    x, y = tensors()
    with pytest.raises(ValueError):
        dispatch.make_launcher(Fake(), spec_of(x, y), cfg)


def test_invalid_config_for_spec_rejected():
    b = aligned(2**21)
    x, y = b[: 2**20], b[2**20:]
    with pytest.raises(ValueError):
        dispatch.make_launcher(Fake(), spec_of(x, y), {"bdim": 128})


def test_workspace_contract():
    x, y = tensors()
    impl = Fake()
    impl.ws_bytes = 64
    L = dispatch.make_launcher(impl, spec_of(x, y))
    with pytest.raises(dispatch.SpecMismatch):
        L([x], [y], None, (), STREAM)
    with pytest.raises(dispatch.SpecMismatch):
        L([x], [y], torch.empty(63, dtype=torch.uint8), (), STREAM)
    with pytest.raises(dispatch.SpecMismatch):
        L([x], [y], torch.empty(64, dtype=torch.int8), (), STREAM)
    L([x], [y], torch.empty(64, dtype=torch.uint8), (), STREAM)
    assert impl.calls[0][2] is not None

    L2 = dispatch.make_launcher(Fake(), spec_of(x, y))
    with pytest.raises(dispatch.SpecMismatch):
        L2([x], [y], torch.empty(64, dtype=torch.uint8), (), STREAM)


def test_bad_workspace_size_rejected():
    x, y = tensors()
    impl = Fake()
    impl.ws_bytes = -1
    with pytest.raises(dispatch.ImplContractError):
        dispatch.make_launcher(impl, spec_of(x, y))


def test_launcher_must_return_none():
    x, y = tensors()
    impl = Fake()
    impl.ret = y
    L = dispatch.make_launcher(impl, spec_of(x, y))
    with pytest.raises(dispatch.ImplContractError):
        L([x], [y], None, (), STREAM)


def test_build_must_return_a_launcher():
    class NoLauncher(Fake):
        def build(self, spec, cfg, ctx):
            return 42

    x, y = tensors()
    with pytest.raises(dispatch.ImplContractError):
        dispatch.make_launcher(NoLauncher(), spec_of(x, y))


def test_launcher_cache_builds_once_per_spec():
    x, y = tensors()
    cache = dispatch.LauncherCache()
    impl = Fake()
    s = spec_of(x, y)
    a = cache.get(impl, s, {"bdim": 256})
    b = cache.get(impl, spec_of(x, y), {"bdim": 256})
    assert a is b and impl.builds == 1
    cache.get(impl, s, {"bdim": 128})
    b2 = aligned()
    cache.get(impl, spec_of(b2[1:33].view(4, 8), y), {"bdim": 256})
    assert impl.builds == 3 and len(cache) == 3


def test_public_call_and_fallback():
    x, y = tensors()
    reg = Registry()
    reg.register(Fake)
    cache = dispatch.LauncherCache()
    dispatch.call("fake", [x], [y], registry=reg, cache=cache, stream=STREAM)
    assert len(cache) == 1

    hit = []
    dispatch.call("missing", [x], [y], registry=reg, cache=cache, fallback=lambda: hit.append(1))
    assert hit == [1]
    with pytest.raises(NotSupported):
        dispatch.call("missing", [x], [y], registry=reg, cache=cache)

    class Refuses(Fake):
        op = "refuses"
        supported = False

    reg.register(Refuses)
    dispatch.call("refuses", [x], [y], registry=reg, cache=cache, fallback=lambda: hit.append(2))
    assert hit == [1, 2] and len(cache) == 1


def test_harness_specs_come_from_callspec_not_tensors():
    # make_launcher takes a CallSpec built anywhere (the harness builds them from samples)
    ts = TensorSpec((4, 8), (8, 1), "f32", 0, 256)
    s = CallSpec("fake", (ts,), (ts,), (), (), (), "cpu")
    L = dispatch.make_launcher(Fake(), s)
    assert L.spec is s
