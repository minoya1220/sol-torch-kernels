"""CPU tests for kernel_hash (design 5.3)."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from sol_kernels._runtime import interface, tuning
from sol_kernels._runtime.interface import Impl
from _load import load_copy_honest

ROOT = Path(__file__).resolve().parents[1]


def test_copy_control_hash_is_stable_hex():
    cls = load_copy_honest().CopyHonest
    h = tuning.kernel_hash(cls)
    assert re.fullmatch(r"[0-9a-f]{64}", h)
    assert tuning.kernel_hash(cls()) == h
    names = [n for n, _ in tuning.kernel_sources(cls)]
    assert names == ["impl", "sol_kernels._runtime.interface"]


def test_hash_is_the_same_in_another_process():
    code = ("import sys; sys.path.insert(0, 'tests'); from _load import load_copy_honest; "
            "from sol_kernels._runtime import tuning; print(tuning.kernel_hash(load_copy_honest().CopyHonest))")
    outs = {
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True,
                       env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert outs == {tuning.kernel_hash(load_copy_honest().CopyHonest)}


# --- a package tree on disk -----------------------------------------------------

TREE = {
    "__init__.py": "from sol_kernels.ops import add, mul\n",
    "_runtime/__init__.py": "",
    "_runtime/interface.py": "INTERFACE_VERSION = 1\n",
    "cute_utils.py": "import cutlass.cute as cute\n",
    "templates/__init__.py": "X = 1\n",
    "templates/elementwise.py": "from . import reduction\nfrom ..cute_utils import cute\n",
    "templates/reduction.py": "import sol_kernels.templates.elementwise\n",  # a cycle
    "templates/unused.py": "",
    "ops/__init__.py": "",
    "ops/add.py": (
        "import math\n"
        "from sol_kernels._runtime.interface import Impl\n"
        "from ..templates import elementwise\n"
        "def f():\n    import sol_kernels.templates.reduction\n"
    ),
    "ops/mul.py": "from sol_kernels.templates.elementwise import cute\n",
}


@pytest.fixture
def pkg(tmp_path, monkeypatch):
    root = tmp_path / "sol_kernels"
    for rel, body in TREE.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def impl_for(rel: str, name: str) -> type:
        mod = types.ModuleType(name)
        mod.__file__ = str(root / rel)
        monkeypatch.setitem(sys.modules, name, mod)
        return type("I", (Impl,), {"op": "x", "__module__": name})

    return root, impl_for


def test_sources_are_the_transitive_sol_kernels_imports(pkg):
    root, impl_for = pkg
    cls = impl_for("ops/add.py", "sol_kernels.ops.add")
    names = [n for n, _ in tuning.kernel_sources(cls, root=root)]
    assert names == [
        "impl",
        "sol_kernels._runtime.interface",
        "sol_kernels.cute_utils",
        "sol_kernels.templates",                 # named by `from ..templates import ...`
        "sol_kernels.templates.elementwise",
        "sol_kernels.templates.reduction",
    ]
    # not followed: sol_kernels/__init__.py (implied parent, which imports every op),
    # sol_kernels.ops, other ops, templates/unused.py, non-sol_kernels modules


def change(root: Path, rel: str, text: str = "\n# edited\n") -> None:
    p = root / rel
    p.write_text(p.read_text() + text)


@pytest.mark.parametrize(
    ("rel", "moves"),
    [
        ("ops/add.py", True),
        ("templates/elementwise.py", True),
        ("templates/reduction.py", True),
        ("templates/__init__.py", True),
        ("cute_utils.py", True),
        ("_runtime/interface.py", True),
        ("templates/unused.py", False),
        ("ops/mul.py", False),
        ("__init__.py", False),
    ],
)
def test_hash_moves_with_exactly_the_hashed_sources(pkg, rel, moves):
    root, impl_for = pkg
    cls = impl_for("ops/add.py", "sol_kernels.ops.add")
    before = tuning.kernel_hash(cls, root=root)
    change(root, rel)
    assert (tuning.kernel_hash(cls, root=root) != before) is moves


def test_hash_moves_with_the_dsl_version_and_interface_version(pkg, monkeypatch):
    root, impl_for = pkg
    cls = impl_for("ops/add.py", "sol_kernels.ops.add")
    before = tuning.kernel_hash(cls, root=root)
    with monkeypatch.context() as m:
        m.setattr(tuning, "dsl_version", lambda: "0.0.0")
        assert tuning.kernel_hash(cls, root=root) != before
    with monkeypatch.context() as m:
        m.setattr(interface, "INTERFACE_VERSION", interface.INTERFACE_VERSION + 1)
        assert tuning.kernel_hash(cls, root=root) != before
    assert tuning.kernel_hash(cls, root=root) == before


def test_entries_are_framed(pkg):
    root, impl_for = pkg
    cls = impl_for("ops/add.py", "sol_kernels.ops.add")
    before = tuning.kernel_hash(cls, root=root)
    # move one byte from the end of one hashed file to the start of the next
    a, b = root / "templates/elementwise.py", root / "templates/reduction.py"
    ta, tb = a.read_text(), b.read_text()
    a.write_text(ta + "#")
    b.write_text(tb)
    one = tuning.kernel_hash(cls, root=root)
    a.write_text(ta)
    b.write_text("#" + tb)
    two = tuning.kernel_hash(cls, root=root)
    assert len({before, one, two}) == 3


def test_missing_dsl_fails_closed(pkg, monkeypatch):
    root, impl_for = pkg
    cls = impl_for("ops/add.py", "sol_kernels.ops.add")

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(tuning.importlib.metadata, "version", missing)
    with pytest.raises(tuning.KernelHashError):
        tuning.kernel_hash(cls, root=root)


def test_missing_or_unparseable_import_fails_closed(pkg):
    root, impl_for = pkg
    (root / "ops/bad.py").write_text("import sol_kernels.templates.nowhere\n")
    with pytest.raises(tuning.KernelHashError):
        tuning.kernel_hash(impl_for("ops/bad.py", "sol_kernels.ops.bad"), root=root)
    (root / "ops/bad2.py").write_text("from .... import x\n")
    with pytest.raises(tuning.KernelHashError):
        tuning.kernel_hash(impl_for("ops/bad2.py", "sol_kernels.ops.bad2"), root=root)
    (root / "templates/elementwise.py").write_text("def (\n")
    with pytest.raises(tuning.KernelHashError):
        tuning.kernel_hash(impl_for("ops/add.py", "sol_kernels.ops.add"), root=root)


def test_hashing_imports_nothing(pkg):
    root, impl_for = pkg
    marker = root.parent / "ran"
    (root / "templates/elementwise.py").write_text(f"open({str(marker)!r}, 'w')\nimport sol_kernels.cute_utils\n")
    tuning.kernel_hash(impl_for("ops/add.py", "sol_kernels.ops.add"), root=root)
    assert not marker.exists()


def test_rejects_non_impls():
    with pytest.raises(TypeError):
        tuning.kernel_hash(object())
