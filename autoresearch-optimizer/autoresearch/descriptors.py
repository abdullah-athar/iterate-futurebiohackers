"""Algorithm descriptors: a short, tiered set of standard terms per program, used to compare strategies.

Each program is described once (cached by normalised-source fingerprint) by one structured LLM call:
1-10 terms, exactly one tier-6 term (what the algorithm fundamentally is), the rest tier 3
(structural choices) or tier 1 (incidental detail). Terms come from a shared vocabulary that only
grows; the prompt asks the model to reuse existing terms wherever one fits.

Distance is a tier-aware soft-chamfer over centred MiniLM term embeddings:
  - every term is matched to its most similar term in the other descriptor, discounted by tier
    disagreement (sqrt(min/max): 6&3 -> 0.71, 6&1 -> 0.41, 3&1 -> 0.58);
  - matches are averaged with fixed tier shares (core 50%, tier-3 35%, tier-1 15%, split evenly
    within a tier, renormalised over the tiers present), symmetrically in both directions.
Identical descriptors are at distance 0. The centring vector `mu` is fixed from SEED_VOCAB so
distances stay comparable as the vocabulary grows.
"""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .novelty import fingerprint

TIER_SHARE = {6: 0.50, 3: 0.35, 1: 0.15}
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fixes the centring vector and is offered to the describer as the starting vocabulary (0 uses).
SEED_PARADIGMS = [
    "simulated annealing", "tabu search", "genetic algorithm", "hill climbing", "iterated local search",
    "beam search", "branch and bound", "dynamic programming", "greedy construction", "random sampling",
    "variable neighbourhood search", "large neighbourhood search", "ant colony optimisation",
    "particle swarm optimisation", "memetic algorithm", "exhaustive enumeration", "integer linear programming",
    "consensus voting", "multiple sequence alignment", "centroid computation", "gradient descent",
    "monte carlo tree search", "constraint propagation", "expectation maximisation", "clustering",
]
SEED_STRUCTURE = [
    "substitution moves", "insertion moves", "deletion moves", "swap moves", "block moves",
    "best-improvement selection", "first-improvement selection", "full neighbourhood scan",
    "random neighbour sampling", "one-point crossover", "uniform crossover", "point mutation",
    "tournament selection", "elitism", "population diversity maintenance", "tabu list",
    "aspiration criterion", "metropolis acceptance", "perturbation kick", "random restarts",
    "stagnation-triggered restart", "prefix extension", "lower-bound pruning", "column-wise majority vote",
    "progressive alignment", "center-star alignment", "edit-operation voting", "edit path traversal",
    "incremental distance update", "delta evaluation", "weighted objective", "set median initialisation",
    "random initialisation", "consensus initialisation", "length adjustment", "multi-start",
    "time-bounded search", "candidate pool", "surrogate scoring", "decomposition into subproblems",
]
SEED_DETAIL = [
    "geometric cooling", "linear cooling", "fixed random seed", "tie-breaking rule", "memoisation",
    "early termination", "shuffled visiting order", "adaptive step size", "fixed iteration cap",
    "best-so-far tracking", "precomputed lookup table",
]
SEED_VOCAB = SEED_PARADIGMS + SEED_STRUCTURE + SEED_DETAIL

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array", "minItems": 1, "maxItems": 10,
            "items": {"type": "object",
                      "properties": {"term": {"type": "string"}, "tier": {"type": "integer", "enum": [1, 3, 6]}},
                      "required": ["term", "tier"]},
        },
        "new_terms": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["terms", "new_terms", "summary"],
}

SYSTEM = "You classify optimisation algorithms with a short standardised vocabulary. Output only the requested JSON."

PROMPT = """Describe the algorithm in the program below with a short set of standard terms, so that it
can be compared with other programs that solve the same problem.

Problem: {problem}

Rules:
- Use 1 to 10 terms, as few as fully characterise the algorithm.
- Exactly one term has tier 6: what the algorithm fundamentally is (its search paradigm).
- Tier 3: structural choices that define how it works (move/neighbourhood types, representation,
  selection or acceptance rule, initialisation when it matters, restart/perturbation mechanism,
  exact subroutines).
- Tier 1: incidental detail (parameter schedules, tie-breaking, caching, bookkeeping).
- Describe the strategy, not the code style. Ignore boilerplate every solver shares: the CPU-time
  deadline check, returning the best string found so far, calling the provided distance metric.
- Terms are short lowercase noun phrases (1-4 words), generic rather than problem-specific.

Vocabulary (term: uses so far), most used first:
{vocab}

Reuse an existing vocabulary term wherever one fits, even if you would have worded it differently.
Introduce a new term only when nothing existing captures the idea, and list every new term in new_terms.
Also give a one-line summary (at most 25 words) of how the algorithm works, as compact pseudocode.

Program:
```python
{source}
```"""


_SPELLING = [("neighbor", "neighbour"), ("ization", "isation"), ("imize", "imise"), ("imizing", "imising")]


def _norm(term: str) -> str:
    """Lowercase, collapse whitespace, British spelling (so 'neighborhood' and 'neighbourhood' are one term)."""
    term = " ".join(term.lower().split())
    for us, uk in _SPELLING:
        term = term.replace(us, uk)
    return term


@dataclass
class Descriptor:
    terms: list[tuple[str, int]]
    new_terms: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def core(self) -> str:
        return next(t for t, tier in self.terms if tier == 6)

    def to_dict(self) -> dict:
        return {"terms": [{"term": t, "tier": tier} for t, tier in self.terms], "new_terms": self.new_terms,
                "summary": self.summary}

    @classmethod
    def from_dict(cls, d: dict) -> Descriptor:
        return cls([(_norm(x["term"]), int(x["tier"])) for x in d["terms"]], [_norm(t) for t in d.get("new_terms", [])],
                   " ".join(d.get("summary", "").split()))

    def __str__(self) -> str:
        return " / ".join(f"{t}[{tier}]" for t, tier in sorted(self.terms, key=lambda x: -x[1]))


class Vocabulary:
    """Term -> usage count. Seed terms start at 0; new terms are appended, nothing is ever pruned."""

    def __init__(self, counts: dict[str, int] | None = None, sizes: list[int] | None = None):
        self.counts = dict(counts) if counts is not None else {t: 0 for t in SEED_VOCAB}
        self.sizes = list(sizes or [])  # vocabulary size logged per generation

    def __len__(self) -> int:
        return len(self.counts)

    def add(self, d: Descriptor) -> list[str]:
        """Count the descriptor's terms; returns the terms that were new to the vocabulary."""
        new = [t for t, _ in d.terms if t not in self.counts]
        for t, _ in d.terms:
            self.counts[t] = self.counts.get(t, 0) + 1
        return new

    def mark_generation(self) -> None:
        self.sizes.append(len(self))

    def prompt_block(self) -> str:
        return "\n".join(f"- {t}: {n}" for t, n in sorted(self.counts.items(), key=lambda x: (-x[1], x[0])))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"counts": self.counts, "sizes": self.sizes}, indent=1))

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        if not path.exists():
            return cls()
        d = json.loads(path.read_text())
        return cls(d["counts"], d.get("sizes"))


def _validate(d: Descriptor) -> str | None:
    if sum(tier == 6 for _, tier in d.terms) != 1:
        return "exactly one term must have tier 6"
    if len({t for t, _ in d.terms}) != len(d.terms):
        return "terms must be distinct"
    return None


_CACHE_LOCK = threading.Lock()


def describe(source: str, vocab: Vocabulary, problem: str, *, model: str = "sonnet",
             cache_path: Path | None = None, use_cache: bool = True, retries: int = 2) -> Descriptor:
    """One structured `claude -p` call per program, cached by normalised-source fingerprint."""
    key = fingerprint(source)
    cache = json.loads(cache_path.read_text()) if cache_path and cache_path.exists() else {}
    if use_cache and key in cache:
        return Descriptor.from_dict(cache[key])
    prompt = PROMPT.format(problem=problem, vocab=vocab.prompt_block(), source=source)
    error = None
    for _ in range(retries + 1):
        text = prompt if error is None else f"{prompt}\n\nYour previous answer was invalid: {error}."
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", model, "--tools", "",
             "--system-prompt", SYSTEM, "--json-schema", json.dumps(DESCRIBE_SCHEMA)],
            input=text, capture_output=True, text=True, timeout=180)
        try:
            out = json.loads(proc.stdout)["structured_output"]
            d = Descriptor.from_dict(out)
            error = _validate(d)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            error = f"unparseable output ({e}): {proc.stderr[-300:]}"
        if error is None:
            break
    else:
        raise RuntimeError(f"describe failed: {error}")
    if cache_path:
        with _CACHE_LOCK:  # re-read under the lock so concurrent describers don't drop each other's entries
            cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
            cache[key] = d.to_dict()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=1))
    return d


class Embedder:
    """MiniLM term embeddings (CPU, cached per term), centred on the fixed SEED_VOCAB mean."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self._raw: dict[str, np.ndarray] = {}
        self.mu = self.raw(SEED_VOCAB).mean(axis=0)

    def raw(self, terms: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(terms) if t not in self._raw]
        if missing:
            for t, e in zip(missing, self.model.encode(missing, normalize_embeddings=True)):
                self._raw[t] = e
        return np.stack([self._raw[t] for t in terms])

    def embed(self, terms: list[str], centred: bool = True) -> np.ndarray:
        E = self.raw(terms)
        if centred:
            E = E - self.mu
            E = E / np.linalg.norm(E, axis=1, keepdims=True)
        return E


def tier_weights(tiers: np.ndarray) -> np.ndarray:
    w = np.array([TIER_SHARE[t] / np.sum(tiers == t) for t in tiers])
    return w / w.sum()


def similarity(a: Descriptor, b: Descriptor, emb: Embedder) -> float:
    ta, tb = np.array([t for _, t in a.terms]), np.array([t for _, t in b.terms])
    S = np.clip(emb.embed([t for t, _ in a.terms]) @ emb.embed([t for t, _ in b.terms]).T, 0.0, 1.0)
    M = S * np.sqrt(np.minimum.outer(ta, tb) / np.maximum.outer(ta, tb))
    return float(0.5 * (tier_weights(ta) @ M.max(axis=1) + tier_weights(tb) @ M.max(axis=0)))


def distance(a: Descriptor, b: Descriptor, emb: Embedder) -> float:
    return 1.0 - similarity(a, b, emb)
