"""The types an impl sees: TensorSpec, CallSpec, Param, Impl, BuildContext (design 6.2, 6.3).

Protected path. Impls may import this module and nothing else from `_runtime`
(design 2.2), so it is pure Python: no torch, no CuTe DSL, no CUDA.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Protocol

INTERFACE_VERSION = 1

# dtype name -> itemsize in bytes. Names are the ones TensorSpec.dtype uses.
DTYPES: Mapping[str, int] = MappingProxyType({
    "bool": 1,
    "u8": 1, "i8": 1,
    "u16": 2, "i16": 2,
    "u32": 4, "i32": 4,
    "u64": 8, "i64": 8,
    "f16": 2, "bf16": 2, "f32": 4, "f64": 8,
})

# Kinds a run-time scalar argument can have: a Python scalar kind or a dtype name.
SCALAR_KINDS = frozenset({"bool", "int", "float"}) | frozenset(DTYPES)

MAX_ALIGN = 256       # TensorSpec.align_bytes is capped here (design 6.2)
MAX_VEC_BYTES = 16    # widest vector access vec_bytes() reports (128-bit loads and stores)


class NotSupported(Exception):
    """Raised by `build` (or the runtime) for a spec the impl does not support."""


class SpecError(ValueError):
    """A TensorSpec or CallSpec that is malformed."""


def _int_tuple(name: str, value: Any) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise SpecError(f"{name} must be a tuple, got {type(value).__name__}")
    for v in value:
        if type(v) is not int:  # bool is an int subclass and is rejected on purpose
            raise SpecError(f"{name} entries must be int, got {v!r}")
    return value


def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    stride: tuple[int, ...]          # in elements; 0 for broadcast dims
    dtype: str                       # a key of DTYPES
    storage_offset: int              # in elements
    align_bytes: int                 # largest power of two (<= MAX_ALIGN) dividing the data pointer

    def __post_init__(self) -> None:
        _int_tuple("shape", self.shape)
        _int_tuple("stride", self.stride)
        if len(self.shape) != len(self.stride):
            raise SpecError(f"shape {self.shape} and stride {self.stride} differ in rank")
        if any(s < 0 for s in self.shape):
            raise SpecError(f"negative size in shape {self.shape}")
        if any(s < 0 for s in self.stride):
            raise SpecError(f"negative stride in {self.stride}")
        if self.dtype not in DTYPES:
            raise SpecError(f"unknown dtype {self.dtype!r}")
        if type(self.storage_offset) is not int or self.storage_offset < 0:
            raise SpecError(f"bad storage_offset {self.storage_offset!r}")
        if type(self.align_bytes) is not int or not _is_pow2(self.align_bytes) or self.align_bytes > MAX_ALIGN:
            raise SpecError(f"align_bytes must be a power of two <= {MAX_ALIGN}, got {self.align_bytes!r}")
        if self.align_bytes < self.itemsize:
            raise SpecError(
                f"data pointer is not aligned to its own element size "
                f"(align_bytes {self.align_bytes} < itemsize {self.itemsize})"
            )

    @property
    def itemsize(self) -> int:
        return DTYPES[self.dtype]

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def is_contiguous(self) -> bool:
        """Row-major contiguous, with torch's rules: dims of size 1 may have any
        stride, and a tensor with no elements is contiguous."""
        if self.numel == 0:
            return True
        expected = 1
        for size, stride in zip(reversed(self.shape), reversed(self.stride)):
            if size != 1 and stride != expected:
                return False
            expected *= size
        return True

    def vec_bytes(self) -> int:
        """Widest vector access (bytes) that is legal along the innermost dim.

        A width w (a power of two, itemsize <= w <= MAX_VEC_BYTES) is legal when the
        innermost dim is unit-stride, w divides the pointer alignment, w divides the
        innermost dim's length in bytes, and w divides every other dim's stride in
        bytes. Then every row starts on a w boundary and splits into whole vectors.
        Anything else, including stride-0 or strided inner dims, rank 0 and empty
        tensors, gets the element size (scalar access).
        """
        isz = self.itemsize
        if self.rank == 0 or self.numel == 0:
            return isz
        inner_size, inner_stride = self.shape[-1], self.stride[-1]
        if inner_size > 1 and inner_stride != 1:
            return isz
        w = min(MAX_VEC_BYTES, self.align_bytes)
        constraints = [inner_size * isz] + [
            st * isz for sz, st in zip(self.shape[:-1], self.stride[:-1]) if sz > 1
        ]
        while w > isz and any(c % w for c in constraints):
            w //= 2
        return max(w, isz)

    def _key(self) -> tuple:
        return (self.shape, self.stride, self.dtype, self.storage_offset, self.align_bytes)


AttrValue = int | float | bool | str


def _typed_attr(value: AttrValue) -> tuple[str, Any]:
    """Attribute value with its type, so True, 1 and 1.0 (equal in Python) and
    0.0 and -0.0 (equal floats) give different CallSpecs."""
    t = type(value)
    if t is bool:
        return ("bool", value)
    if t is int:
        return ("int", value)
    if t is float:
        if not math.isfinite(value):
            raise SpecError(f"attribute value must be finite, got {value!r}")
        return ("float", value.hex())
    if t is str:
        return ("str", value)
    raise SpecError(f"attribute value must be int, float, bool or str, got {t.__name__}")


@dataclass(frozen=True, eq=False)
class CallSpec:
    """One concrete call's static description. Hashable; the launcher cache key.

    `attrs` is stored sorted by name. Equality and hashing include each attribute
    value's type, so `keepdim=True` and `keepdim=1` are different specs.
    """

    op: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    attrs: tuple[tuple[str, AttrValue], ...]   # compile-time attributes: dim, keepdim, ...
    scalar_kinds: tuple[str, ...]              # run-time scalar args: their kinds, not values
    aliases: tuple[tuple[int, int], ...]       # (output index, input index) pairs that share memory
    device: str                                # the tuning device key (design 5.2)

    def __post_init__(self) -> None:
        if not isinstance(self.op, str) or not self.op:
            raise SpecError("op must be a non-empty str")
        for name, group in (("inputs", self.inputs), ("outputs", self.outputs)):
            if not isinstance(group, tuple) or not all(isinstance(t, TensorSpec) for t in group):
                raise SpecError(f"{name} must be a tuple of TensorSpec")
        if not isinstance(self.attrs, tuple):
            raise SpecError("attrs must be a tuple of (name, value) pairs")
        names = []
        for pair in self.attrs:
            if not (isinstance(pair, tuple) and len(pair) == 2 and isinstance(pair[0], str) and pair[0]):
                raise SpecError(f"bad attribute {pair!r}")
            _typed_attr(pair[1])
            names.append(pair[0])
        if len(set(names)) != len(names):
            raise SpecError(f"duplicate attribute names in {names}")
        object.__setattr__(self, "attrs", tuple(sorted(self.attrs, key=lambda p: p[0])))
        if not isinstance(self.scalar_kinds, tuple) or not all(k in SCALAR_KINDS for k in self.scalar_kinds):
            raise SpecError(f"scalar_kinds must be a tuple of {sorted(SCALAR_KINDS)}")
        if not isinstance(self.aliases, tuple):
            raise SpecError("aliases must be a tuple of (output index, input index) pairs")
        for pair in self.aliases:
            if not (isinstance(pair, tuple) and len(pair) == 2 and all(type(i) is int for i in pair)):
                raise SpecError(f"bad alias {pair!r}")
            o, i = pair
            if not (0 <= o < len(self.outputs) and 0 <= i < len(self.inputs)):
                raise SpecError(f"alias {pair} out of range")
        if len(set(self.aliases)) != len(self.aliases):
            raise SpecError("duplicate alias pairs")
        object.__setattr__(self, "aliases", tuple(sorted(self.aliases)))
        if not isinstance(self.device, str) or not self.device:
            raise SpecError("device must be a non-empty str")

    def attr(self, name: str) -> AttrValue:
        for k, v in self.attrs:
            if k == name:
                return v
        raise KeyError(name)

    def _key(self) -> tuple:
        return (
            self.op,
            tuple(t._key() for t in self.inputs),
            tuple(t._key() for t in self.outputs),
            tuple((k, _typed_attr(v)) for k, v in self.attrs),
            self.scalar_kinds,
            self.aliases,
            self.device,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CallSpec):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def digest(self) -> str:
        """sha256 of a canonical encoding. Unlike hash(), stable across processes
        and machines, so reports and logs can name a spec."""
        blob = json.dumps(self._key(), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(blob.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class Param:
    options: tuple[int, ...]
    default: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.options, tuple) or not self.options:
            raise SpecError("Param.options must be a non-empty tuple")
        _int_tuple("Param.options", self.options)
        if len(set(self.options)) != len(self.options):
            raise SpecError(f"duplicate options in {self.options}")
        if self.default is not None and self.default not in self.options:
            raise SpecError(f"default {self.default!r} not in options {self.options}")


Config = dict[str, int]

# Launcher(inputs: tuple[cute.Tensor, ...], outputs: tuple[cute.Tensor, ...],
#          workspace: cute.Tensor | None, scalars: tuple, stream) -> None
Launcher = Callable[..., None]


class BuildContext(Protocol):
    """Provided by the runtime to `Impl.build` (implemented in dispatch.py)."""

    def compile(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """sol_compile: compiles through the CuTe DSL and records kernel names
        and module hashes for the allowlist (design 2.3)."""
        ...

    def dummies(self) -> tuple[tuple[Any, ...], tuple[Any, ...], Any]:
        """(inputs, outputs, workspace) as cute tensors with the spec's layout,
        poisoned, for compilation only. Freed after build."""
        ...

    def stream(self) -> Any:
        """A placeholder stream to compile the launcher's stream argument against.
        Not in design 6.3, which gives build no way to name the stream type."""
        ...


class Impl:
    """Base class for kernel impls (design 6.3). See the contract there."""

    op: ClassVar[str]
    search_space: ClassVar[dict[str, Param]] = {}

    def supports(self, spec: CallSpec) -> bool:
        raise NotImplementedError

    def tune_key(self, spec: CallSpec) -> dict[str, int]:
        return {}

    def default_config(self, spec: CallSpec) -> Config:
        return {
            name: (p.default if p.default is not None else p.options[0])
            for name, p in self.search_space.items()
        }

    def valid_config(self, spec: CallSpec, cfg: Config) -> bool:
        return True

    def workspace_bytes(self, spec: CallSpec, cfg: Config) -> int:
        return 0

    def build(self, spec: CallSpec, cfg: Config, ctx: BuildContext) -> Launcher:
        raise NotImplementedError


class Registry:
    """op name -> Impl subclass. One impl per op."""

    def __init__(self) -> None:
        self._impls: dict[str, type[Impl]] = {}

    def register(self, cls: type[Impl]) -> type[Impl]:
        if not (isinstance(cls, type) and issubclass(cls, Impl)) or cls is Impl:
            raise TypeError(f"register_impl needs an Impl subclass, got {cls!r}")
        op = getattr(cls, "op", None)
        if not isinstance(op, str) or not op:
            raise TypeError(f"{cls.__qualname__} has no op name")
        if op in self._impls:
            raise ValueError(f"op {op!r} already has impl {self._impls[op].__qualname__}")
        self._impls[op] = cls
        return cls

    def get(self, op: str) -> type[Impl] | None:
        return self._impls.get(op)

    def ops(self) -> tuple[str, ...]:
        return tuple(sorted(self._impls))


REGISTRY = Registry()


def register_impl(cls: type[Impl]) -> type[Impl]:
    """Class decorator: adds the impl to the registry under cls.op."""
    return REGISTRY.register(cls)
