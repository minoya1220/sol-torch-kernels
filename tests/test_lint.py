"""CPU tests for the static lint (design 2.2) and the flag catalogue (2.6, 9.5)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from sol_eval.flags import CheckResult, Flag, Severity
from sol_eval.integrity import lint

ROOT = Path(__file__).resolve().parents[1]
I, C, F = "FORBIDDEN_IMPORT", "FORBIDDEN_CALL", "FORBIDDEN_FILE"


def found(src: str, **kw) -> set[tuple[str, str]]:
    return {(r.code.code, r.evidence["rule"]) for r in lint.lint_source(src, path="t.py", **kw)}


# --- the flag catalogue ------------------------------------------------------


def test_flag_catalogue_matches_design_table():
    text = (ROOT / "docs" / "eval-design.md").read_text()
    table = text.split("### 2.6 Flag catalogue", 1)[1].split("Every flag instance", 1)[0]
    rows = re.findall(r"^\| (`[A-Z0-9_]+`(?:, `[A-Z0-9_]+`)*) \| ([A-Z]+)", table, re.M)
    design = {code.strip("`"): sev for codes, sev in rows for code in codes.split(", ")}
    assert design and set(design) == {f.code for f in Flag}
    for code, sev in design.items():
        assert Flag[code].severity is Severity(sev), code
    assert Flag.TUNING_STALE.ci_severity is Severity.FAIL and Flag.TUNING_STALE.severity is Severity.WARN


def test_check_result_rejects_unknown_evidence_keys():
    with pytest.raises(ValueError):
        CheckResult(Flag.FORBIDDEN_CALL, False, None, {"file": "x", "secret": 1})
    with pytest.raises(TypeError):
        CheckResult("FORBIDDEN_CALL", False)  # type: ignore[arg-type]


# --- rules, one snippet each --------------------------------------------------

CHEATS = {
    # imports outside the allowlist
    "import torch": {(I, "import")},
    "import cuda.bindings.driver as d": {(I, "import")},
    "from torch import add": {(I, "import")},
    "import ctypes, os, sys, subprocess, threading, importlib, builtins, gc, inspect, pickle, socket, time, random, numpy": {(I, "import")},
    "import cutlass.utils.layout": {(I, "import")},
    "import cutlass.base_dsl": {(I, "import")},
    "from __future__ import annotations": {(I, "import")},
    "from math import *": {(I, "import")},
    "from . import x": {(I, "import")},
    "from cuda.bindings import driver": {(I, "import")},
    "from cutlass import does_not_exist_xyz": {(I, "import")},
    "from cutlass.cute.runtime import ctypes": {(I, "import")},
    "from cutlass import base_dsl": {(I, "import")},
    "from sol_kernels._runtime.interface import json": {(I, "import")},
    "from sol_kernels._runtime import dispatch": {(I, "import")},
    "from sol_kernels import call": {(I, "import")},
    "import sol_kernels": {(I, "import")},
    # banned modules reached through attributes of allowed ones
    "import cutlass.utils as u\nu.importlib.import_module('torch')": {(I, "module_via_attribute")},
    "import enum\nenum.bltns.print(1)": {(I, "module_via_attribute")},
    "import typing\ntyping.sys.modules": {(I, "module_via_attribute")},
    "import dataclasses\ndataclasses.inspect.stack()": {(I, "module_via_attribute")},
    "import cutlass.cute.runtime as rt\nrt.ctypes.CDLL": {(I, "module_via_attribute")},
    "import cutlass\ncutlass.base_dsl": {(I, "module_via_attribute")},
    "import sol_kernels._runtime.interface as i\ni.hashlib.md5": {(I, "module_via_attribute")},
    "import sol_kernels._runtime.interface\nsol_kernels._runtime.dispatch": {(I, "module_via_attribute")},
    "import typing\nt = typing\nt.sys": {(I, "module_via_attribute")},
    "import typing\ngetattr(typing, 'sys')": {(I, "module_via_attribute")},
    # banned builtins, used or referenced
    "eval('1')": {(C, "banned_builtin")},
    "exec('x = 1')": {(C, "banned_builtin")},
    "compile('1', 'f', 'eval')": {(C, "banned_builtin")},
    "open('/etc/passwd')": {(C, "banned_builtin")},
    "globals()": {(C, "banned_builtin")},
    "vars()": {(C, "banned_builtin")},
    "f = eval": {(C, "banned_builtin")},
    "import functools\nfunctools.partial(open, 'x')": {(C, "banned_builtin")},
    "__import__('os')": {(C, "banned_builtin")},
    # dynamic attribute access
    "getattr(x, name)": {(C, "dynamic_attribute")},
    "setattr(x, name, 1)": {(C, "dynamic_attribute")},
    "delattr(x, name)": {(C, "dynamic_attribute")},
    "getattr(x, *a)": {(C, "dynamic_attribute")},
    "g = getattr": {(C, "dynamic_attribute")},
    "import operator\noperator.attrgetter(n)": {(C, "dynamic_attribute")},
    "import operator\noperator.methodcaller(n)": {(C, "dynamic_attribute")},
    "from operator import attrgetter as a\nf = a": {(C, "dynamic_attribute")},
    # dunders
    "x.__class__": {(C, "dunder")},
    "f.__globals__['os']": {(C, "dunder")},
    "x.__name__": {(C, "dunder")},
    "super().__init__()": {(C, "dunder")},
    "__builtins__": {(C, "dunder")},
    "getattr(x, '__dict__')": {(C, "dunder")},
    "import operator\noperator.attrgetter('a.__globals__')": {(C, "dunder")},
    "import operator\noperator.methodcaller('__setattr__', 'x', 1)": {(C, "dunder")},
    "'{0.__globals__}'.format(f)": {(C, "dunder")},
    "def __repr__(self): pass": {(C, "dunder")},
    "def __getattr__(name): pass": {(C, "dunder")},
    "class __X__: pass": {(C, "dunder")},
    "def f(__x__): pass": {(C, "dunder")},
    "f(__x__=1)": {(C, "dunder")},
    "match x:\n    case A(__dict__=d): pass": {(C, "dunder")},
    "import math as __m__": {(C, "dunder")},
    # assignment to imported modules and objects
    "import math\nmath.pi = 3": {(C, "module_attribute_assignment")},
    "import math\nmath.pi += 1": {(C, "module_attribute_assignment")},
    "import math\ndel math.pi": {(C, "module_attribute_assignment")},
    "import math\nmath.x: int = 1": {(C, "module_attribute_assignment")},
    "import math\nsetattr(math, 'pi', 3)": {(C, "module_attribute_assignment")},
    "import math\ndelattr(math, 'pi')": {(C, "module_attribute_assignment")},
    "import math\nm = math\nm.pi = 3": {(C, "module_attribute_assignment")},
    "import math\na, b = math, 1\na.pi = 3": {(C, "module_attribute_assignment")},
    "import math\nfor math.pi in range(3): pass": {(C, "module_attribute_assignment")},
    "import functools\nfunctools.WRAPPER_ASSIGNMENTS[0] = 'x'": {(C, "module_attribute_assignment")},
    "from cutlass.cute import Tensor\nTensor.x = 1": {(C, "module_attribute_assignment")},
    "import cutlass.cute as cute\nc = cute.arch\nc.thread_idx = None": {(C, "module_attribute_assignment")},
    "def f():\n    import math\n    math.pi = 3": {(C, "module_attribute_assignment")},
    # global
    "def f():\n    global X": {(C, "global_statement")},
    # kernel names and decorator use
    "import cutlass.cute as cute\n@cute.kernel\ndef copy(x): pass": {(C, "kernel_name")},
    "from cutlass.cute import kernel\n@kernel\ndef copy(x): pass": {(C, "kernel_name")},
    "from cutlass import cute as c\n@c.kernel\ndef copy(x): pass": {(C, "kernel_name")},
    "import cutlass.cute as cute\nk = cute.kernel(f)": {(C, "kernel_decorator")},
    "import cutlass.cute as cute\nk = cute.kernel\n@k\ndef sol_x(x): pass": {(C, "kernel_decorator")},
    # files that cannot be linted
    "def (": {(F, "unparseable")},
    "x = 1\0": {(F, "unparseable")},
}


@pytest.mark.parametrize("src", list(CHEATS), ids=lambda s: s.splitlines()[-1][:50])
def test_cheat_is_flagged(src):
    assert found(src) == CHEATS[src]


HONEST = [
    "import cutlass\nimport cutlass.cute as cute\nx = cutlass.Int32",
    "from cutlass import cute\ny = cute.arch.thread_idx",
    "import cutlass.cute.runtime\ncutlass.cute.runtime.from_dlpack",
    "import cutlass.utils",
    "import math, operator, functools, itertools, dataclasses, typing, enum",
    "from dataclasses import dataclass, field\n@dataclass(frozen=True)\nclass P:\n    n: int = field(default=1)\n    def __post_init__(self): pass",
    "from typing import Any\nfrom enum import Enum\nclass K(Enum):\n    A = 1",
    "import functools\n@functools.cache\ndef f(n): return n",
    "from sol_kernels._runtime.interface import CallSpec, Config, Impl, Param",
    "class A:\n    def __init__(self):\n        self.x = 1\n    def __call__(self): return self.x",
    "getattr(x, 'y')\ngetattr(x, 'y', None)\nsetattr(obj, 'y', 1)",
    "import operator\nf = operator.attrgetter('a.b')\ng = operator.methodcaller('m', 1)",
    "import cutlass.cute as cute\n@cute.kernel\ndef sol_copy(x): pass\n@cute.jit\ndef host(x): sol_copy(x)",
    "class A:\n    x = 1\nA.x = 2\na = A()\na.y = 3",
    "def f():\n    '''Calls __init__ and reads {0.__dict__}.'''\n    return 1",
    "import math\nmath = 3",
    "x = math_like.pi\nx.y = 1",
]


@pytest.mark.parametrize("src", HONEST, ids=lambda s: s.splitlines()[-1][:50])
def test_honest_code_passes(src):
    assert found(src) == set()


def test_honest_copy_control_passes():
    assert lint.lint_file(ROOT / "eval/exploits/_controls/copy_honest/impl.py") == []


def test_evidence_names_file_line_and_rule():
    (r,) = lint.lint_source("x = 1\nimport torch\n", path="ops/add.py")
    assert r.code is Flag.FORBIDDEN_IMPORT and not r.passed and r.case_id is None
    assert r.evidence == {"file": "ops/add.py", "line": 2, "col": 0, "rule": "import",
                          "detail": "import torch", "module": "torch"}
    assert r.message() == "ops/add.py:2:0: FORBIDDEN_IMPORT [import] import torch"


def test_one_flag_per_problem_not_per_node():
    rs = lint.lint_source("import cutlass.cute.runtime as rt\nrt.ctypes.CDLL.x.y\n", path="t.py")
    assert len(rs) == 1


# --- a kernel tree: relative imports, templates, cute_utils, files -----------


def make_tree(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body if isinstance(body, bytes) else body.encode())
    return tmp_path


BASE = {
    "src/sol_kernels/__init__.py": "import torch\nfrom sol_kernels._runtime import dispatch\n",
    "src/sol_kernels/_runtime/__init__.py": "",
    "src/sol_kernels/_runtime/interface.py": "import json\nclass Impl: pass\n",
    "src/sol_kernels/_runtime/dispatch.py": "import torch\n",
    "src/sol_kernels/templates/__init__.py": "",
    "src/sol_kernels/templates/elementwise.py": "import cutlass.cute as cute\nimport cutlass.cute.runtime as rt\ndef unary(): pass\n",
    "src/sol_kernels/cute_utils.py": "import cutlass.cute as cute\nfrom cutlass.cute.runtime import ctypes\n",
    "src/sol_kernels/ops/__init__.py": "",
}


def tree_results(root: Path, rel: str | None = None) -> set[tuple[str, str, str]]:
    rs = lint.lint_tree(root) if rel is None else lint.lint_file(root / rel, repo_root=root)
    return {(r.evidence["file"], r.code.code, r.evidence["rule"]) for r in rs}


def test_tree_lint_scope_and_kernel_module_resolution(tmp_path):
    root = make_tree(tmp_path, {**BASE, "src/sol_kernels/ops/add.py": (
        "from ..templates import elementwise\n"
        "from sol_kernels.templates import elementwise as e2\n"
        "import sol_kernels.templates.elementwise\n"
        "from .. import cute_utils\n"
        "from sol_kernels._runtime.interface import Impl\n"
        "elementwise.unary\n"
        "e2.cute.arch\n"
        "sol_kernels.templates.elementwise.unary\n"
    )})
    assert tree_results(root) == {
        ("src/sol_kernels/cute_utils.py", I, "import"),  # its own `from ... import ctypes`
    }


@pytest.mark.parametrize(
    ("line", "rule"),
    [
        ("from ..templates import elementwise\nelementwise.rt.ctypes", "module_via_attribute"),
        ("from sol_kernels.cute_utils import ctypes", "import"),
        ("from .. import cute_utils\ncute_utils.ctypes.CDLL", "module_via_attribute"),
        ("import sol_kernels.templates.elementwise\nsol_kernels.torch", "module_via_attribute"),
        ("from ..ops import other", "import"),
        ("from ...outside import x", "import"),
        ("from sol_kernels._runtime.interface import json", "import"),
    ],
)
def test_tree_reexports_from_kernel_modules_are_caught(tmp_path, line, rule):
    root = make_tree(tmp_path, {**BASE, "src/sol_kernels/ops/add.py": line, "src/sol_kernels/ops/other.py": ""})
    assert ("src/sol_kernels/ops/add.py", I, rule) in tree_results(root, "src/sol_kernels/ops/add.py")


def test_kernel_code_is_never_imported_by_the_lint(tmp_path):
    marker = tmp_path / "ran"
    root = make_tree(tmp_path, {
        **BASE,
        "src/sol_kernels/cute_utils.py": f"open({str(marker)!r}, 'w')\nimport cutlass.cute as cute\n",
        "src/sol_kernels/ops/add.py": "from sol_kernels.cute_utils import cute\ncute.arch\n",
    })
    tree_results(root)
    assert not marker.exists()
    assert not any(m == "sol_kernels.cute_utils" for m in sys.modules)


def test_tree_rejects_files_that_are_not_plain_python(tmp_path):
    root = make_tree(tmp_path, {
        **BASE,
        "src/sol_kernels/ops/kernel.cubin": b"\x7fELF",
        "src/sol_kernels/ops/helper.so": b"\x7fELF",
        "src/sol_kernels/ops/conftest.py": "",
        "src/sol_kernels/templates/__main__.py": "",
        "src/sol_kernels/ops/__pycache__/add.cpython-313.pyc": b"\0",
    })
    (root / "src/sol_kernels/ops/link.py").symlink_to(root / "src/sol_kernels/_runtime/dispatch.py")
    got = tree_results(root)
    assert ("src/sol_kernels/ops/kernel.cubin", F, "non_python_file") in got
    assert ("src/sol_kernels/ops/helper.so", F, "non_python_file") in got
    assert ("src/sol_kernels/ops/conftest.py", F, "autorun") in got
    assert ("src/sol_kernels/templates/__main__.py", F, "autorun") in got
    assert ("src/sol_kernels/ops/__pycache__/add.cpython-313.pyc", F, "non_python_file") in got
    assert ("src/sol_kernels/ops/link.py", F, "symlink") in got


def test_in_git_ignored_caches_are_skipped_but_committed_ones_are_not(tmp_path):
    root = make_tree(tmp_path, {**BASE, ".gitignore": "__pycache__/\n*.pyc\n",
                                "src/sol_kernels/ops/add.py": "import math\n",
                                "src/sol_kernels/ops/__pycache__/add.cpython-313.pyc": b"\0",
                                "src/sol_kernels/ops/new.py": "import torch\n"})
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", ".gitignore", "src/sol_kernels/ops/add.py"], check=True)
    got = tree_results(root)
    assert got == {("src/sol_kernels/cute_utils.py", I, "import"), ("src/sol_kernels/ops/new.py", I, "import")}
    subprocess.run(git + ["add", "-f", "src/sol_kernels/ops/__pycache__/add.cpython-313.pyc"], check=True)
    assert ("src/sol_kernels/ops/__pycache__/add.cpython-313.pyc", F, "non_python_file") in tree_results(root)


def test_pr_file_rules_match_the_protected_paths_check():
    rs = lint.check_pr_files(["src/sol_kernels/ops/add.py", "sitecustomize.py", "eval/x/conftest.py", "a/b.pth"])
    assert {(r.evidence["file"], r.evidence["rule"]) for r in rs} == {
        ("sitecustomize.py", "autorun_outside_harness"),
        ("eval/x/conftest.py", "autorun_in_kernel_pr"),
        ("a/b.pth", "autorun_outside_harness"),
    }
    assert lint.check_pr_files(["src/sol_kernels/ops/add.py", "tuning/add.csv"]) == []
    assert lint.check_pr_files(["tests/conftest.py"]) == []  # a harness PR: protected, not forbidden


# --- the CLI that CI runs -----------------------------------------------------


def test_cli_on_the_repo_passes():
    p = subprocess.run([sys.executable, "-m", "sol_eval.integrity.lint"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "0 problem(s)" in p.stdout


def test_cli_fails_on_a_cheat():
    f = ROOT / "eval/exploits/E0001-torch-fallback/impl.py"
    p = subprocess.run([sys.executable, "-m", "sol_eval.integrity.lint", str(f)], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 1
    assert "eval/exploits/E0001-torch-fallback/impl.py:7:0: FORBIDDEN_IMPORT [import] import torch" in p.stdout


def test_ci_runs_the_lint():
    wf = (ROOT / ".github/workflows/ci.yml").read_text()
    lint_job = wf.split("  lint:", 1)[1].split("\n  unit:", 1)[0]
    assert "uv sync --frozen" in lint_job and "uv run python -m sol_eval.integrity.lint" in lint_job
    assert "Placeholder" not in wf
