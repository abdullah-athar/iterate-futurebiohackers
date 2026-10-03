"""Static guard run before any evaluation: candidates may only import what the problem allows.

This is not a sandbox (the subprocess + timeout in sandbox.py is the containment layer); it is
a cheap, interpretable check that stops a proposer from scoring itself, reading the benchmark
generator, shelling out, or pulling in packages the evaluation environment does not have.
"""

from __future__ import annotations

import ast
import sys

DENIED_STDLIB = {
    "os", "sys", "subprocess", "shutil", "pathlib", "importlib", "socket", "ctypes", "multiprocessing",
    "threading", "signal", "inspect", "builtins", "pickle", "marshal", "code", "runpy", "glob", "tempfile",
}
DENIED_CALLS = {"__import__", "eval", "exec", "open", "compile", "globals", "vars"}


def _allowed(module: str, allowed: tuple[str, ...]) -> bool:
    root = module.split(".")[0]
    if any(module == a or module.startswith(a + ".") for a in allowed):
        return True
    return root in sys.stdlib_module_names and root not in DENIED_STDLIB


def check_imports(source: str, allowed: tuple[str, ...]) -> list[str]:
    """Return a list of violations (empty = OK). Syntax errors are left for the evaluator to report."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if not _allowed(a.name, allowed)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or not _allowed(mod, allowed):
                bad.append(f"from {'.' * node.level}{mod} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in DENIED_CALLS:
            bad.append(f"call to {node.func.id}()")
    return sorted(set(bad))
