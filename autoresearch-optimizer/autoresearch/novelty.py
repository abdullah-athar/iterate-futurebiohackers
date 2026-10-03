"""Novelty gate: rejection sampling of near-duplicate proposals before spending an evaluation.

Sources are normalised (docstrings/comments removed, AST re-printed) so that cosmetic edits
do not count as new ideas. Exact matches and sources whose similarity ratio to a prior
candidate exceeds `threshold` are rejected.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import hashlib
from dataclasses import dataclass

_KEEP = set(dir(builtins)) | {"solve", "instance", "self"}


def normalize(source: str) -> str:
    """Strip docstrings/comments and alpha-rename local identifiers so renames are not 'novel'."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "\n".join(l.strip() for l in source.splitlines() if l.strip() and not l.strip().startswith("#"))
    keep = set(_KEEP)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.update((a.asname or a.name).split(".")[0] for a in node.names)
    rename: dict[str, str] = {}

    def canon(name: str) -> str:
        if name in keep:
            return name
        return rename.setdefault(name, f"_v{len(rename)}")

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
        if isinstance(node, ast.Name):
            node.id = canon(node.id)
        elif isinstance(node, ast.arg):
            node.arg = canon(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "solve":
            node.name = canon(node.name)
    return ast.unparse(tree)


def fingerprint(source: str) -> str:
    return hashlib.sha1(normalize(source).encode()).hexdigest()[:12]


def similarity(a_norm: str, b_norm: str) -> float:
    m = difflib.SequenceMatcher(None, a_norm, b_norm, autojunk=False)
    if m.real_quick_ratio() < 0.5:
        return m.real_quick_ratio()
    return m.ratio()


@dataclass
class NoveltyVerdict:
    fingerprint: str
    max_similarity: float
    nearest_id: int | None
    is_duplicate: bool

    def to_dict(self) -> dict:
        return {"fingerprint": self.fingerprint, "max_similarity": round(self.max_similarity, 4),
                "nearest_id": self.nearest_id, "is_duplicate": self.is_duplicate}


def check_novelty(source: str, prior: list[tuple[int, str]], threshold: float = 0.95) -> NoveltyVerdict:
    """`prior` is a list of (candidate_id, raw_source) for everything already proposed."""
    norm = normalize(source)
    fp = hashlib.sha1(norm.encode()).hexdigest()[:12]
    best_sim, best_id = 0.0, None
    for cid, src in prior:
        other = normalize(src)
        if other == norm:
            return NoveltyVerdict(fp, 1.0, cid, True)
        sim = similarity(norm, other)
        if sim > best_sim:
            best_sim, best_id = sim, cid
    return NoveltyVerdict(fp, best_sim, best_id, best_sim >= threshold)
