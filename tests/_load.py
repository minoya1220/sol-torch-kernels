"""Load an impl file by path, the way the exploit runner (design 9.3) will."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_module(rel_path: str, name: str) -> ModuleType:
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_copy_honest() -> ModuleType:
    if "_controls_copy_honest" in sys.modules:
        return sys.modules["_controls_copy_honest"]
    return load_module("eval/exploits/_controls/copy_honest/impl.py", "_controls_copy_honest")
