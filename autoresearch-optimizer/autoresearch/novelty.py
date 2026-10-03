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
import re
from collections import Counter
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


@dataclass
class NoveltyVerdict:
    fingerprint: str
    max_similarity: float
    nearest_id: int | None
    is_duplicate: bool

    def to_dict(self) -> dict:
        return {"fingerprint": self.fingerprint, "max_similarity": round(self.max_similarity, 4),
                "nearest_id": self.nearest_id, "is_duplicate": self.is_duplicate}


_TOKEN = re.compile(r"\w+|[^\w\s]")
_FEATURES: dict[str, tuple[str, list[str], Counter]] = {}


def _features(source: str) -> tuple[str, list[str], Counter]:
    """Normalised text, its token sequence and token counts (cached by source hash)."""
    key = hashlib.sha1(source.encode()).hexdigest()
    if key not in _FEATURES:
        norm = normalize(source)
        toks = _TOKEN.findall(norm)
        _FEATURES[key] = (norm, toks, Counter(toks))
    return _FEATURES[key]


def check_novelty(source: str, prior: list[tuple[int, str]], threshold: float = 0.95, exact_top: int = 5) -> NoveltyVerdict:
    """`prior` is a list of (candidate_id, raw_source) for everything already proposed.

    Similarity is difflib's ratio over *token* sequences of the normalised sources (character-level
    matching is ~100x slower on multi-KB solvers). The token-multiset overlap is a cheap upper
    bound on that ratio, so the exact ratio is only computed for the `exact_top` closest candidates
    by the bound (and any whose bound reaches the threshold).
    """
    norm, toks, counts = _features(source)
    fp = hashlib.sha1(norm.encode()).hexdigest()[:12]
    bounds = []
    for cid, src in prior:
        other, otoks, ocounts = _features(src)
        if other == norm:
            return NoveltyVerdict(fp, 1.0, cid, True)
        bounds.append((2 * sum((counts & ocounts).values()) / max(len(toks) + len(otoks), 1), cid, otoks))
    bounds.sort(key=lambda b: -b[0])
    best_sim, best_id = 0.0, None
    for i, (bound, cid, otoks) in enumerate(bounds):
        if bound <= best_sim or (i >= exact_top and bound < threshold):
            break
        sim = difflib.SequenceMatcher(None, toks, otoks, autojunk=False).ratio()
        if sim > best_sim:
            best_sim, best_id = sim, cid
    return NoveltyVerdict(fp, best_sim, best_id, best_sim >= threshold)
