"""Static lint of impl files (design 2.2). Runs in CI on CPU.

Lints every file under `src/sol_kernels/ops/`, `src/sol_kernels/templates/` and
`src/sol_kernels/cute_utils.py`, and any impl file passed by path (the exploit
corpus). Each problem is a `CheckResult` with flag `FORBIDDEN_IMPORT`,
`FORBIDDEN_CALL` or `FORBIDDEN_FILE` and evidence naming the file, line and rule.

The lint is a tripwire, not the boundary (2.0, 2.5). Its job is to make the honest
path the easy one and to fail loudly on accident. Where the design's rules can be
read more than one way, this module takes the stricter reading; each such choice
is marked "stricter reading" below.

Rules (2.2):

- `FORBIDDEN_IMPORT`
  - an import of a module outside the allowlist (`ALLOWED_EXACT`, `ALLOWED_PREFIXES`);
  - a star import, a `__future__` import, or a relative import that cannot be
    resolved (stricter reading: none of these is on the allowlist);
  - `from X import n` where `n` is a module outside the allowlist, or cannot be
    resolved;
  - an attribute chain from an imported name that reaches a module outside the
    allowlist, for example `cutlass.utils.importlib` or `enum.bltns` (stricter
    reading: allowed modules re-export banned ones, and reaching a module by
    attribute is importing it).
- `FORBIDDEN_CALL`
  - any use of `__import__`, `eval`, `exec`, `compile`, `open`, `globals`, `vars`;
  - `getattr`, `setattr`, `delattr` other than as a direct call with a string
    literal attribute name, and `operator.attrgetter` or `operator.methodcaller`
    other than with string literals (they are `getattr` by another name);
  - any identifier, attribute or literal attribute name that starts and ends
    with `__`, other than a `def` of `__init__`, `__call__`, `__post_init__` or
    `__name__`, including `{0.__globals__}`-style format fields (stricter
    reading: reading `x.__name__` or calling `super().__init__()` is also
    flagged, since only definitions are excepted);
  - assignment, augmented assignment or deletion of an attribute or item of an
    imported name, or of a name bound to one (`ev = torch.cuda.Event`), and
    `setattr` or `delattr` on one (stricter reading: names imported with
    `from X import obj` count, since patching them patches the library);
  - `global` statements;
  - a `@cute.kernel` function whose name does not start with `sol_`, and any use
    of `cute.kernel` other than as a decorator, which would dodge the naming rule.
    2.6 has no code for the naming rule; it is reported as `FORBIDDEN_CALL`.
- `FORBIDDEN_FILE`
  - in the linted trees: any file that is not a `.py` file, a symlink, an auto-run
    file name, or a file that does not parse;
  - in a PR's changed paths (`check_pr_files`): the H3 rules shared with the
    `protected-paths` check (`paths.forbidden_files`).

Names are resolved without importing kernel code. Modules under `sol_kernels` are
read from their files with `ast`. Allowlisted modules outside `sol_kernels`
(`cutlass`, the listed stdlib modules) are trusted, pinned dependencies; the lint
imports them to see what their attributes are.

Known limits, for the dynamic checks (2.2 audit hooks and tripwire) to cover:
attribute chains through call results (`f().x`), function arguments and
containers are not followed; mutation through methods (`m.registry.append`) is
not caught.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import inspect
import re
import subprocess
import sys
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sol_eval.flags import CheckResult, Flag
from sol_eval.integrity import paths as pathrules

REPO_ROOT = Path(__file__).resolve().parents[3]

# What CI lints, relative to the repo root. Design 2.2 names ops/ and templates/;
# cute_utils.py is added (stricter reading) because impls may import it and kernel
# PRs may edit it (8.2), so leaving it out would be a way around the lint.
LINT_SCOPE: tuple[str, ...] = (
    "src/sol_kernels/ops",
    "src/sol_kernels/templates",
    "src/sol_kernels/cute_utils.py",
)

ALLOWED_EXACT = frozenset({
    "cutlass",
    "cutlass.cute",
    "cutlass.utils",   # exact: 2.2 lists submodules only for cutlass.cute
    "math", "operator", "functools", "itertools", "dataclasses", "typing", "enum",
    "sol_kernels.cute_utils",
    "sol_kernels._runtime.interface",
})
ALLOWED_PREFIXES = ("cutlass.cute.", "sol_kernels.templates.")

BANNED_NAMES = frozenset({"__import__", "eval", "exec", "compile", "open", "globals", "vars"})
ATTR_BUILTINS = {"getattr": (2, 3), "setattr": (3,), "delattr": (2,)}
DUNDER_DEFS = frozenset({"__init__", "__call__", "__post_init__", "__name__"})
KERNEL_PREFIX = "sol_"

_DUNDER = re.compile(r"^__\w+__$")
_FORMAT_DUNDER = re.compile(r"\{[^{}]*\.__\w+__[^{}]*\}")


def module_allowed(name: str) -> bool:
    return name in ALLOWED_EXACT or name.startswith(ALLOWED_PREFIXES)


def _ancestors() -> frozenset[str]:
    out = {p.rstrip(".") for p in ALLOWED_PREFIXES}
    for n in list(ALLOWED_EXACT) + list(out):
        parts = n.split(".")
        out.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return frozenset(out - ALLOWED_EXACT)


# Packages that an allowed import passes through (`import sol_kernels.templates.x`
# binds `sol_kernels`). Walking an attribute chain through them is fine; binding
# them directly by import is not.
ANCESTORS = _ancestors()


def _is_dunder(name: str) -> bool:
    return bool(_DUNDER.match(name))


# ---------------------------------------------------------------------------
# name resolution


@dataclass(frozen=True)
class Ref:
    """What a name or attribute chain refers to."""

    kind: str          # "module", "object" (a trusted object) or "local" (opaque)
    name: str = ""     # module name, for kind "module"
    obj: Any = None    # the object, for kind "object"


_MISSING = object()


class Resolver:
    """Resolves module attributes. `sol_kernels` modules are read from source and
    never imported; allowlisted modules outside `sol_kernels` are imported."""

    def __init__(self, kernel_root: Path | None = None) -> None:
        self.kernel_root = kernel_root or REPO_ROOT / "src" / "sol_kernels"
        self._bindings: dict[str, dict[str, tuple]] = {}
        self._active: set[tuple[str, str]] = set()

    # sol_kernels modules ----------------------------------------------------

    @staticmethod
    def is_package_source(name: str) -> bool:
        return name == "sol_kernels" or name.startswith("sol_kernels.")

    def source_file(self, name: str) -> Path | None:
        if not self.is_package_source(name):
            return None
        p = self.kernel_root.joinpath(*name.split(".")[1:])
        if p.is_dir() and (p / "__init__.py").is_file():
            return p / "__init__.py"
        f = p.with_name(p.name + ".py") if name != "sol_kernels" else p / "__init__.py"
        return f if f.is_file() else None

    def bindings(self, name: str) -> dict[str, tuple]:
        """Names a sol_kernels module binds by import, anywhere in the file."""
        if name not in self._bindings:
            self._bindings[name] = {}
            f = self.source_file(name)
            if f is not None:
                try:
                    tree = ast.parse(f.read_bytes(), str(f))
                except (SyntaxError, ValueError):
                    tree = None
                if tree is not None:
                    package = name if f.name == "__init__.py" else name.rpartition(".")[0]
                    self._bindings[name] = import_bindings(tree, package)
        return self._bindings[name]

    # trusted modules --------------------------------------------------------

    def load(self, name: str) -> types.ModuleType | None:
        if self.is_package_source(name) or not module_allowed(name):
            raise AssertionError(f"refusing to import {name!r} in the lint")
        try:
            return importlib.import_module(name)
        except Exception:
            return None

    # resolution -------------------------------------------------------------

    def binding(self, spec: tuple) -> Ref | None:
        if spec[0] == "module":
            return Ref("module", spec[1])
        if spec[0] == "from":
            return self.module_attr(spec[1], spec[2])
        return None  # unresolvable relative import

    def module_attr(self, modname: str, attr: str) -> Ref | None:
        """None means the attribute cannot be resolved."""
        if self.is_package_source(modname):
            key = (modname, attr)
            if key in self._active:
                return Ref("local")
            spec = self.bindings(modname).get(attr)
            if spec is not None:
                self._active.add(key)
                try:
                    return self.binding(spec)
                finally:
                    self._active.discard(key)
            if self.source_file(f"{modname}.{attr}") is not None:
                return Ref("module", f"{modname}.{attr}")
            return Ref("local") if self.source_file(modname) is not None else None
        if not module_allowed(modname):
            return None
        mod = self.load(modname)
        if mod is None:
            return None
        val = getattr(mod, attr, _MISSING)
        if val is _MISSING:
            sub = f"{modname}.{attr}"
            if not module_allowed(sub):
                return None
            try:
                val = importlib.import_module(sub)
            except Exception:
                return None
        if isinstance(val, types.ModuleType):
            return Ref("module", val.__name__)
        return Ref("object", obj=val)

    def attr(self, ref: Ref, attr: str) -> Ref | None:
        if ref.kind == "module":
            return self.module_attr(ref.name, attr)
        if ref.kind == "object":
            try:
                val = inspect.getattr_static(ref.obj, attr, _MISSING)
            except Exception:
                return None
            if val is _MISSING:
                return None
            if isinstance(val, types.ModuleType):
                return Ref("module", val.__name__)
            return Ref("object", obj=val)
        return None

    def trusted_object(self, modname: str, attr: str) -> Any:
        ref = self.module_attr(modname, attr)
        return ref.obj if ref is not None and ref.kind == "object" else _MISSING


def resolve_relative(module: str | None, level: int, package: str | None) -> str | None:
    if level == 0:
        return module
    if not package:
        return None
    parts = package.split(".")
    if level - 1 >= len(parts):
        return None
    base = ".".join(parts[: len(parts) - (level - 1)])
    return f"{base}.{module}" if module else base


def import_bindings(tree: ast.AST, package: str | None) -> dict[str, tuple]:
    """Map each name an import binds (in any scope) to ("module", name),
    ("from", module, attr) or ("unresolved",). A name bound twice keeps the last."""
    out: dict[str, tuple] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = ("module", a.name)
                else:
                    out[a.name.split(".")[0]] = ("module", a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = resolve_relative(node.module, node.level, package)
            for a in node.names:
                if a.name == "*":
                    continue
                out[a.asname or a.name] = ("from", mod, a.name) if mod else ("unresolved",)
    return out


# ---------------------------------------------------------------------------
# the lint


def _chain(expr: ast.AST) -> tuple[str, list[str]] | None:
    """`a.b.c` or `getattr(a, "b").c` -> ("a", ["b", "c"])."""
    attrs: list[str] = []
    while True:
        if isinstance(expr, ast.Attribute):
            attrs.append(expr.attr)
            expr = expr.value
        elif (
            isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "getattr"
            and len(expr.args) >= 2 and isinstance(expr.args[1], ast.Constant) and isinstance(expr.args[1].value, str)
        ):
            attrs.append(expr.args[1].value)
            expr = expr.args[0]
        elif isinstance(expr, ast.Subscript):
            attrs.append("")  # an item: opaque, but still rooted at the name
            expr = expr.value
        elif isinstance(expr, ast.Name):
            return expr.id, attrs[::-1]
        else:
            return None


@dataclass
class _Linter(ast.NodeVisitor):
    path: str
    package: str | None
    resolver: Resolver
    results: list[CheckResult] = field(default_factory=list)
    names: dict[str, list[Ref]] = field(default_factory=dict)  # import-derived names
    decorators: set[int] = field(default_factory=set)
    _seen: set[tuple] = field(default_factory=set)

    # reporting --------------------------------------------------------------

    def flag(self, code: Flag, node: ast.AST, rule: str, detail: str, **extra: Any) -> None:
        key = (code, getattr(node, "lineno", 0), getattr(node, "col_offset", 0), rule, detail)
        if key in self._seen:
            return
        self._seen.add(key)
        ev = {"file": self.path, "line": getattr(node, "lineno", 0), "col": getattr(node, "col_offset", 0),
              "rule": rule, "detail": detail, **extra}
        self.results.append(CheckResult(code, False, None, ev))

    # setup ------------------------------------------------------------------

    def prepare(self, tree: ast.Module) -> None:
        for name, spec in import_bindings(tree, self.package).items():
            ref = self.resolver.binding(spec)
            self.names[name] = [ref] if ref is not None else [Ref("local")]
        for node in ast.walk(tree):
            for fn in getattr(node, "decorator_list", ()):
                self.decorators.add(id(fn.func if isinstance(fn, ast.Call) else fn))
        # Names assigned from an import-derived name or attribute chain are
        # import-derived too (flow-insensitive, to a fixed point).
        for _ in range(20):
            changed = False
            for node in ast.walk(tree):
                for target, value in _simple_assignments(node):
                    c = _chain(value)
                    if c is None or c[0] not in self.names:
                        continue
                    refs = [r for r in (self.walk_refs(c) or [Ref("local")])]
                    have = self.names.setdefault(target, [])
                    for r in refs:
                        if r not in have:
                            have.append(r)
                            changed = True
            if not changed:
                break

    def walk_refs(self, c: tuple[str, list[str]]) -> list[Ref]:
        out = []
        for ref in self.names.get(c[0], []):
            r: Ref | None = ref
            for a in c[1]:
                if r is None or a == "":
                    r = None
                    break
                r = self.resolver.attr(r, a)
            if r is not None:
                out.append(r)
        return out

    # imports ----------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if not module_allowed(a.name):
                self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"import {a.name}", module=a.name)
            if a.asname and _is_dunder(a.asname):
                self.flag(Flag.FORBIDDEN_CALL, node, "dunder", a.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = resolve_relative(node.module, node.level, self.package)
        shown = "." * node.level + (node.module or "")
        if node.module == "__future__":
            self.flag(Flag.FORBIDDEN_IMPORT, node, "import", "from __future__ import ...", module="__future__")
            return
        if mod is None:
            self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"cannot resolve relative import from {shown!r}", module=shown)
            return
        for a in node.names:
            if a.name == "*":
                self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"from {mod} import *", module=mod)
                continue
            for n in (a.name, a.asname):
                if n and _is_dunder(n):
                    self.flag(Flag.FORBIDDEN_CALL, node, "dunder", n)
            if not module_allowed(mod) and not Resolver.is_package_source(mod):
                self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"from {mod} import {a.name}", module=mod)
                continue
            ref = self.resolver.module_attr(mod, a.name)
            if ref is None:
                self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"cannot resolve {mod}.{a.name}", module=f"{mod}.{a.name}")
            elif ref.kind == "module":
                if not module_allowed(ref.name):
                    self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"from {mod} import {a.name} is module {ref.name}", module=ref.name)
            elif not module_allowed(mod):
                self.flag(Flag.FORBIDDEN_IMPORT, node, "import", f"from {mod} import {a.name}", module=mod)
        self.generic_visit(node)

    # names, attributes, calls -----------------------------------------------

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in BANNED_NAMES:
            self.flag(Flag.FORBIDDEN_CALL, node, "banned_builtin", node.id)
        elif node.id in ATTR_BUILTINS:
            self.flag(Flag.FORBIDDEN_CALL, node, "dynamic_attribute", f"{node.id} used other than as a call with a literal name")
        elif _is_dunder(node.id):
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.id)
        else:
            self._check_chain(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr):
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.attr)
        self._check_chain(node)
        # Report the chain once at its outermost node; still visit what it is
        # rooted on when that is not a plain name.
        inner = node.value
        while isinstance(inner, ast.Attribute):
            if _is_dunder(inner.attr):
                self.flag(Flag.FORBIDDEN_CALL, inner, "dunder", inner.attr)
            inner = inner.value
        if not isinstance(inner, ast.Name):
            self.visit(inner)
        elif inner.id in BANNED_NAMES or inner.id in ATTR_BUILTINS or _is_dunder(inner.id):
            self.visit_Name(inner)

    def _check_chain(self, node: ast.AST) -> None:
        c = _chain(node)
        if c is None or c[0] not in self.names:
            return
        for ref in self.names[c[0]]:
            r: Ref | None = ref
            path = c[0]
            for a in c[1]:
                if r is None or a == "":
                    r = None
                    break
                r = self.resolver.attr(r, a)
                path += "." + a
                if r is not None and r.kind == "module" and not (module_allowed(r.name) or r.name in ANCESTORS):
                    self.flag(Flag.FORBIDDEN_IMPORT, node, "module_via_attribute", f"{path} is module {r.name}", module=r.name)
                    r = None
                    break
            if r is not None and r.kind == "object":
                self._check_special_object(node, r.obj)

    def _check_special_object(self, node: ast.AST, obj: Any) -> None:
        if obj is self.resolver.trusted_object("cutlass.cute", "kernel") and id(node) not in self.decorators:
            self.flag(Flag.FORBIDDEN_CALL, node, "kernel_decorator", "cute.kernel used other than as a decorator")
        op = (self.resolver.trusted_object("operator", "attrgetter"), self.resolver.trusted_object("operator", "methodcaller"))
        if obj in op and not getattr(node, "_sol_literal_call", False):
            self.flag(Flag.FORBIDDEN_CALL, node, "dynamic_attribute",
                      "operator.attrgetter or methodcaller used other than as a call with literal names")

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if isinstance(f, ast.Name) and f.id in ATTR_BUILTINS:
            arities = ATTR_BUILTINS[f.id]
            starred = any(isinstance(a, ast.Starred) for a in node.args) or node.keywords
            lit = len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str)
            if starred or len(node.args) not in arities or not lit:
                self.flag(Flag.FORBIDDEN_CALL, node, "dynamic_attribute", f"{f.id} without a literal attribute name")
            else:
                if _is_dunder(node.args[1].value):
                    self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.args[1].value)
                if f.id in ("setattr", "delattr"):
                    self._check_mutation_target(node.args[0], node, f"{f.id} on")
            if f.id != "getattr":
                for a in node.args:
                    self.visit(a)
                return
            # getattr(x, "lit"): the chain check runs on the call as a whole
            self._check_chain(node)
            for a in node.args:
                self.visit(a)
            return
        self._check_operator_call(node)
        self.generic_visit(node)

    def _check_operator_call(self, node: ast.Call) -> None:
        c = _chain(node.func)
        if c is None or c[0] not in self.names:
            return
        ag = self.resolver.trusted_object("operator", "attrgetter")
        mc = self.resolver.trusted_object("operator", "methodcaller")
        for r in self.walk_refs(c):
            if r.kind != "object" or r.obj not in (ag, mc):
                continue
            args = node.args if r.obj is ag else node.args[:1]
            ok = (
                bool(args) and not node.keywords if r.obj is ag else bool(args)
            ) and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in args)
            ok = ok and not any(isinstance(a, ast.Starred) for a in node.args)
            if ok:
                for a in args:
                    for seg in a.value.split("."):
                        if _is_dunder(seg):
                            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", seg)
                node.func._sol_literal_call = True  # type: ignore[attr-defined]
            else:
                self.flag(Flag.FORBIDDEN_CALL, node, "dynamic_attribute",
                          "operator.attrgetter or methodcaller without literal names")
                node.func._sol_literal_call = True  # type: ignore[attr-defined]  # flagged once here

    # assignments ------------------------------------------------------------

    def _check_mutation_target(self, target: ast.AST, node: ast.AST, how: str) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for e in target.elts:
                self._check_mutation_target(e, node, how)
            return
        if isinstance(target, ast.Starred):
            self._check_mutation_target(target.value, node, how)
            return
        if how == "assignment" and not isinstance(target, (ast.Attribute, ast.Subscript)):
            return
        c = _chain(target)
        if c is not None and c[0] in self.names:
            self.flag(Flag.FORBIDDEN_CALL, node, "module_attribute_assignment",
                      f"{how} {ast.unparse(target)}, which belongs to imported {c[0]!r}")

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._check_mutation_target(t, node, "assignment")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._check_mutation_target(node.target, node, "assignment")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_mutation_target(node.target, node, "assignment")
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        for t in node.targets:
            self._check_mutation_target(t, node, "assignment")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._check_mutation_target(node.target, node, "assignment")
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_withitem(self, node: ast.withitem) -> None:
        if node.optional_vars is not None:
            self._check_mutation_target(node.optional_vars, node.context_expr, "assignment")
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._check_mutation_target(node.target, node.iter, "assignment")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.flag(Flag.FORBIDDEN_CALL, node, "global_statement", "global " + ", ".join(node.names))

    # definitions ------------------------------------------------------------

    def _check_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if _is_dunder(node.name) and node.name not in DUNDER_DEFS:
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.name)
        kernel = self.resolver.trusted_object("cutlass.cute", "kernel")
        for d in node.decorator_list:
            target = d.func if isinstance(d, ast.Call) else d
            c = _chain(target)
            if c is None or c[0] not in self.names:
                continue
            if any(r.kind == "object" and r.obj is kernel for r in self.walk_refs(c)) \
                    and not node.name.startswith(KERNEL_PREFIX):
                self.flag(Flag.FORBIDDEN_CALL, node, "kernel_name",
                          f"@cute.kernel function {node.name!r} must start with {KERNEL_PREFIX!r}")
        for a in _all_args(node.args):
            if _is_dunder(a.arg):
                self.flag(Flag.FORBIDDEN_CALL, a, "dunder", a.arg)
        self.generic_visit(node)

    visit_FunctionDef = _check_def
    visit_AsyncFunctionDef = _check_def

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for a in _all_args(node.args):
            if _is_dunder(a.arg):
                self.flag(Flag.FORBIDDEN_CALL, a, "dunder", a.arg)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if _is_dunder(node.name):
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.name)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg and _is_dunder(node.arg):
            self.flag(Flag.FORBIDDEN_CALL, node.value, "dunder", node.arg)
        self.generic_visit(node)

    def visit_MatchClass(self, node: ast.MatchClass) -> None:
        for a in node.kwd_attrs:
            if _is_dunder(a):
                self.flag(Flag.FORBIDDEN_CALL, node, "dunder", a)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name and _is_dunder(node.name):
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", node.name)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _FORMAT_DUNDER.search(node.value):
            self.flag(Flag.FORBIDDEN_CALL, node, "dunder", "format field reads a dunder attribute")


def _all_args(a: ast.arguments) -> Iterator[ast.arg]:
    yield from a.posonlyargs
    yield from a.args
    yield from a.kwonlyargs
    if a.vararg:
        yield a.vararg
    if a.kwarg:
        yield a.kwarg


def _simple_assignments(node: ast.AST) -> Iterator[tuple[str, ast.AST]]:
    def pairs(target: ast.AST, value: ast.AST) -> Iterator[tuple[str, ast.AST]]:
        if isinstance(target, ast.Name):
            yield target.id, value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            for t, v in zip(target.elts, value.elts):
                yield from pairs(t, v)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for t in target.elts:
                if isinstance(t, ast.Name):
                    yield t.id, value
    if isinstance(node, ast.Assign):
        for t in node.targets:
            yield from pairs(t, node.value)
    elif isinstance(node, ast.AnnAssign) and node.value is not None:
        yield from pairs(node.target, node.value)
    elif isinstance(node, ast.NamedExpr):
        yield from pairs(node.target, node.value)
    elif isinstance(node, (ast.For, ast.AsyncFor)):
        yield from pairs(node.target, node.iter)
    elif isinstance(node, ast.withitem) and node.optional_vars is not None:
        yield from pairs(node.optional_vars, node.context_expr)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                out.add(id(body[0].value))
    return out


def module_name_for(path: Path, src_root: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(src_root.resolve())
    except ValueError:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or None


def lint_source(
    source: bytes | str,
    *,
    path: str,
    module_name: str | None = None,
    is_package: bool = False,
    resolver: Resolver | None = None,
) -> list[CheckResult]:
    """Lint one file. `module_name` is its dotted name when it is part of
    `sol_kernels` (needed for relative imports); None for files outside it."""
    try:
        tree = ast.parse(source, path)
    except (SyntaxError, ValueError) as e:
        return [CheckResult(Flag.FORBIDDEN_FILE, False, None,
                            {"file": path, "rule": "unparseable", "detail": f"{type(e).__name__}: {e}"})]
    package = None
    if module_name:
        package = module_name if is_package else module_name.rpartition(".")[0] or None
    linter = _Linter(path=path, package=package, resolver=resolver or Resolver())
    linter.prepare(tree)
    docstrings = _docstring_nodes(tree)
    original = linter.visit_Constant
    linter.visit_Constant = lambda n: None if id(n) in docstrings else original(n)  # type: ignore[method-assign]
    linter.visit(tree)
    return sorted(linter.results, key=lambda r: (r.evidence.get("line", 0), r.evidence.get("col", 0), r.code.code))


def lint_file(path: Path, *, repo_root: Path = REPO_ROOT, resolver: Resolver | None = None) -> list[CheckResult]:
    """Lint one file by path, inside or outside `sol_kernels`."""
    shown = _rel(path, repo_root)
    if path.is_symlink():
        return [CheckResult(Flag.FORBIDDEN_FILE, False, None,
                            {"file": shown, "rule": "symlink", "detail": "symlinks are not allowed in impl trees"})]
    name = module_name_for(path, repo_root / "src")
    if name is not None and not Resolver.is_package_source(name):
        name = None
    return lint_source(path.read_bytes(), path=shown, module_name=name,
                       is_package=path.name == "__init__.py",
                       resolver=resolver or Resolver(repo_root / "src" / "sol_kernels"))


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix() if not path.is_symlink() \
            else path.absolute().relative_to(root.absolute()).as_posix()
    except ValueError:
        return path.as_posix()


def scope_files(repo_root: Path = REPO_ROOT, scope: Iterable[str] = LINT_SCOPE) -> list[Path]:
    """Files in the lint scope. In a git work tree: tracked files plus untracked
    ones that are not ignored (so a committed `.pyc` is seen and a local
    `__pycache__` is not). Elsewhere: every file on disk."""
    scope = list(scope)
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *scope],
            check=True, capture_output=True,
        ).stdout
        files = sorted({repo_root / p for p in out.decode().split("\0") if p})
        return [f for f in files if f.exists() or f.is_symlink()]
    except (OSError, subprocess.CalledProcessError):
        found: list[Path] = []
        for s in scope:
            p = repo_root / s
            if p.is_file() or p.is_symlink():
                found.append(p)
            elif p.is_dir():
                found.extend(sorted(q for q in p.rglob("*") if q.is_file() or q.is_symlink()))
        return found


def lint_tree(repo_root: Path = REPO_ROOT, scope: Iterable[str] = LINT_SCOPE) -> list[CheckResult]:
    resolver = Resolver(repo_root / "src" / "sol_kernels")
    results: list[CheckResult] = []
    for f in scope_files(repo_root, scope):
        shown = _rel(f, repo_root)
        if pathrules.is_autorun_name(shown):
            results.append(CheckResult(Flag.FORBIDDEN_FILE, False, None,
                                       {"file": shown, "rule": "autorun", "detail": "auto-executed file name"}))
        elif f.is_symlink() or f.suffix != ".py":
            rule = "symlink" if f.is_symlink() else "non_python_file"
            results.append(CheckResult(Flag.FORBIDDEN_FILE, False, None,
                                       {"file": shown, "rule": rule, "detail": "impl trees hold only .py files"}))
        else:
            results.extend(lint_file(f, repo_root=repo_root, resolver=resolver))
    return results


def check_pr_files(paths: Iterable[str], protected_globs: tuple[str, ...] | None = None) -> list[CheckResult]:
    """`FORBIDDEN_FILE` for a PR's changed paths (H3), with the same rules as the
    `protected-paths` check."""
    globs = protected_globs or pathrules.load_protected_globs(REPO_ROOT / ".github" / "protected-paths")
    return [
        CheckResult(Flag.FORBIDDEN_FILE, False, None, {"file": p, "rule": rule, "detail": "auto-executed file"})
        for p, rule in pathrules.forbidden_files(list(paths), globs)
    ]


def lint_exploit(entry: Path, *, repo_root: Path = REPO_ROOT) -> list[CheckResult]:
    """The `lint` layer of one exploit-corpus entry (9.2): the lint of its
    `impl.py`, if it has one, plus the PR-path rules on its `[[pr_files]]`. The
    impl file is only parsed, never imported."""
    import tomllib

    meta = tomllib.loads((entry / "exploit.toml").read_text(encoding="utf-8"))
    results: list[CheckResult] = []
    impl = entry / "impl.py"
    if impl.exists():
        results.extend(lint_file(impl, repo_root=repo_root))
    pr_paths = [f["path"] for f in meta.get("pr_files", [])]
    if pr_paths:
        results.extend(check_pr_files(pr_paths))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sol_eval.integrity.lint", description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", type=Path, help="files to lint (default: the impl trees)")
    ap.add_argument("--root", type=Path, default=REPO_ROOT, help="repo root")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if args.files:
        resolver = Resolver(root / "src" / "sol_kernels")
        results = [r for f in args.files for r in lint_file(f, repo_root=root, resolver=resolver)]
        checked = len(args.files)
    else:
        results = lint_tree(root)
        checked = len(scope_files(root))
    for r in results:
        print(r.message())
    print(f"lint: {checked} file(s), {len(results)} problem(s)")
    return 1 if results else 0


if __name__ == "__main__":
    sys.exit(main())
