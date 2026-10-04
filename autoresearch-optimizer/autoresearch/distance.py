"""Distances between signatures behind one interface (diversity extension).

    metric.distance(a, b) -> DistanceResult(value | None, metric, version, available, reason, components)
    nearest(sig, refs, metric) -> (DistanceResult, ref id)

weighted_jaccard (default; an interpretable reference, not claimed optimal):
    d(A, B) = 1 - sum_t min(wA(t), wB(t)) / sum_t max(wA(t), wB(t))
    wX(t) = role weight of term t in X (paradigm 6, mechanism 3, detail 1, unknown term 1), 0 if absent.
    In [0, 1], symmetric, 0 iff both have the same weighted term set; synonyms are mapped to one id before
    this, so a rewording does not move it. Example: same paradigm and mechanisms, one detail more -> 1/(w+1).

minilm (the existing backend, Johann's): 1 - tier-aware soft-chamfer similarity of centred MiniLM term
    embeddings (autoresearch/descriptors.py). Audit: embeddings are re-normalised after centring and cosines
    clipped to [0, 1] before the tier-weighted average, so the similarity is in [0, 1] and the distance too;
    it is not the 1 - cos in [0, 2] case, and nothing is rescaled. It is symmetric (both match directions are
    averaged) and 0 for identical descriptors. Its centring set is part of its version.

A missing, empty or invalid signature makes the distance *unavailable* (value None): it is information we
do not have, never evidence of a duplicate. Semantic proximity is a separate notion from exact duplication,
which the code novelty gate decides on the normalised source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .families import Signature
from .json_cache import text_hash


@dataclass
class DistanceResult:
    value: float | None
    metric: str
    version: str
    available: bool
    reason: str = ""
    components: dict = field(default_factory=dict)   # shared / only_a / only_b terms with their weights

    def to_dict(self) -> dict:
        return {"value": None if self.value is None else round(self.value, 4), "metric": self.metric,
                "version": self.version, "available": self.available, "reason": self.reason,
                "components": self.components}


class WeightedJaccard:
    name = "weighted_jaccard"

    def __init__(self, role_weights: dict[str, float]) -> None:
        self.role_weights = role_weights
        self.version = f"wjaccard-v1:{text_hash(str(sorted(role_weights.items())))}"

    def distance(self, a: Signature | None, b: Signature | None) -> DistanceResult:
        if a is None or b is None or a.empty or b.empty:
            return DistanceResult(None, self.name, self.version, False, "missing or empty signature")
        wa, wb = a.weights(self.role_weights), b.weights(self.role_weights)
        keys = set(wa) | set(wb)
        den = sum(max(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
        if den <= 0:
            return DistanceResult(None, self.name, self.version, False, "zero total weight")
        num = sum(min(wa.get(k, 0.0), wb.get(k, 0.0)) for k in keys)
        d = 1.0 - num / den
        comps = {"shared": sorted(k for k in keys if k in wa and k in wb),
                 "only_a": sorted(k for k in wa if k not in wb), "only_b": sorted(k for k in wb if k not in wa)}
        return DistanceResult(min(1.0, max(0.0, d)), self.name, self.version, True, components=comps)


class MiniLMDistance:
    """Johann's descriptor distance, available only for signatures of the existing backend."""

    name = "minilm"

    def __init__(self) -> None:
        from .descriptors import SEED_VOCAB, Embedder
        self._emb: Embedder | None = None
        self._embedder_cls = Embedder
        self.version = f"minilm-centred-v1:{text_hash(' '.join(SEED_VOCAB))}"

    @property
    def emb(self):
        if self._emb is None:
            self._emb = self._embedder_cls()
        return self._emb

    def distance(self, a: Signature | None, b: Signature | None) -> DistanceResult:
        from .descriptors import Descriptor, distance
        if not (a and b and a.raw.get("terms") and b.raw.get("terms")):
            return DistanceResult(None, self.name, self.version, False, "needs existing-backend descriptors")
        d = distance(Descriptor.from_dict(a.raw), Descriptor.from_dict(b.raw), self.emb)
        if not math.isfinite(d):
            return DistanceResult(None, self.name, self.version, False, "non-finite distance")
        return DistanceResult(min(1.0, max(0.0, d)), self.name, self.version, True)


def get_metric(name: str, role_weights: dict[str, float]):
    if name == "weighted_jaccard":
        return WeightedJaccard(role_weights)
    if name == "minilm":
        return MiniLMDistance()
    raise ValueError(f"unknown distance metric {name!r} (weighted_jaccard | minilm)")


def nearest(sig: Signature | None, refs: list[tuple[object, Signature | None]], metric) -> tuple[DistanceResult, object]:
    """Closest available reference; (unavailable result, None) when no distance can be computed."""
    best, best_id = None, None
    for rid, ref in refs:
        r = metric.distance(sig, ref)
        if r.available and (best is None or r.value < best.value):
            best, best_id = r, rid
    if best is None:
        return DistanceResult(None, metric.name, metric.version, False, "no comparable reference"), None
    return best, best_id
