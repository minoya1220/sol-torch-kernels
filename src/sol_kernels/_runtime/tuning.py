"""Tuning support (design 5.3). Holds `kernel_hash`; the TuningTable lookup comes
with the tuner (implementation step 10).

Protected path.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import inspect
from pathlib import Path
from typing import Any

from . import interface

PACKAGE_ROOT = Path(__file__).resolve().parents[1]   # src/sol_kernels
PACKAGE = "sol_kernels"
DSL_DISTRIBUTION = "nvidia-cutlass-dsl"


class KernelHashError(RuntimeError):
    """The hash cannot be computed, so no tuning row can be trusted for the impl."""


def _source_file(name: str, root: Path) -> Path | None:
    """File of a `sol_kernels` module, found without importing anything."""
    parts = name.split(".")[1:]
    p = root.joinpath(*parts)
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if parts and p.with_name(p.name + ".py").is_file():
        return p.with_name(p.name + ".py")
    return None


def _in_package(name: str) -> bool:
    return name == PACKAGE or name.startswith(PACKAGE + ".")


def _resolve(module: str | None, level: int, package: str | None, where: str) -> str:
    if level == 0:
        return module or ""
    parts = (package or "").split(".") if package else []
    if not parts or level - 1 >= len(parts):
        raise KernelHashError(f"{where}: cannot resolve a relative import of level {level}")
    base = ".".join(parts[: len(parts) - (level - 1)])
    return f"{base}.{module}" if module else base


def imported_modules(source: bytes, *, path: str, package: str | None, root: Path = PACKAGE_ROOT) -> set[str]:
    """`sol_kernels` modules named by the import statements of one file, in any
    scope. For `from X import n`, X and, when it is a submodule, X.n. Parent
    packages implied by a dotted name are not added: design 5.3 says the modules
    are found by scanning import statements, and `sol_kernels/__init__.py`, which
    every such import runs, will import every op."""
    try:
        tree = ast.parse(source, path)
    except (SyntaxError, ValueError) as e:
        raise KernelHashError(f"{path}: does not parse: {e}") from e
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if _in_package(a.name))
        elif isinstance(node, ast.ImportFrom):
            mod = _resolve(node.module, node.level, package, path)
            if not _in_package(mod):
                continue
            out.add(mod)
            for a in node.names:
                sub = f"{mod}.{a.name}"
                if a.name != "*" and _source_file(sub, root) is not None:
                    out.add(sub)
    return out


def _impl_file(impl: Any) -> tuple[Path, str]:
    cls = impl if isinstance(impl, type) else type(impl)
    if not (isinstance(cls, type) and issubclass(cls, interface.Impl)):
        raise TypeError(f"{impl!r} is not an Impl")
    try:
        f = inspect.getsourcefile(cls)
    except TypeError:
        f = None
    if not f or not Path(f).is_file():
        raise KernelHashError(f"{cls.__qualname__}: no source file")
    return Path(f), cls.__module__


def kernel_sources(impl: Any, *, root: Path = PACKAGE_ROOT) -> list[tuple[str, bytes]]:
    """What the hash covers, in hash order: the impl's own module, then every
    `sol_kernels` module it imports, transitively, sorted by name."""
    f, modname = _impl_file(impl)
    first = f.read_bytes()
    own = modname if _in_package(modname) else None
    package = (modname if f.name == "__init__.py" else modname.rpartition(".")[0]) if own else None

    seen: dict[str, bytes] = {}
    todo = sorted(imported_modules(first, path=str(f), package=package, root=root))
    while todo:
        name = todo.pop()
        if name in seen or name == own:
            continue
        sf = _source_file(name, root)
        if sf is None:
            raise KernelHashError(f"{modname} imports {name}, which has no source under {root}")
        data = sf.read_bytes()
        seen[name] = data
        pkg = name if sf.name == "__init__.py" else name.rpartition(".")[0]
        todo.extend(sorted(imported_modules(data, path=str(sf), package=pkg, root=root) - seen.keys()))
    return [("impl", first)] + sorted(seen.items())


def dsl_version() -> str:
    try:
        return importlib.metadata.version(DSL_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as e:
        raise KernelHashError(f"{DSL_DISTRIBUTION} is not installed") from e


def kernel_hash(impl: Any, *, root: Path = PACKAGE_ROOT) -> str:
    """sha256, as 64 hex digits, over (design 5.3, in this order): the impl
    module's source bytes; the source bytes of every `sol_kernels` module it
    imports, transitively; the installed nvidia-cutlass-dsl version;
    `interface.INTERFACE_VERSION`. Each entry is framed with a tag and its length,
    so moving bytes between entries changes the hash. Reads files only: nothing is
    imported, so computing a hash runs no impl code."""
    h = hashlib.sha256()

    def put(tag: str, data: bytes) -> None:
        h.update(tag.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)

    for name, data in kernel_sources(impl, root=root):
        put("module:" + name, data)
    put("dsl", dsl_version().encode())
    put("interface", str(interface.INTERFACE_VERSION).encode())
    return h.hexdigest()
