"""Programs, lineages, families and the canonical vocabulary (diversity extension).

Four distinct notions:
  program   a ledger entry: candidate file + run config + provenance (parent_ids, mode, generation).
  lineage   genealogy only: an entry inherits the lineage of its first parent, except that a
            `new_family` proposal (or the seed) founds a new lineage. Merges keep their other parents in
            the ledger; lineage never looks at descriptors.
  family    an explicit algorithmic class: the canonical paradigm id of the program's signature. The LLM
            proposes a signature from the code; this module maps it to vocabulary ids and assigns the
            family. Two lineages can share a family; a paradigm change moves a program to another family
            without touching its lineage.
  proposal  a planned change before code generation (autoresearch/planner.py).

Family rule (deterministic): family = canonical paradigm id. A hybrid gets one principal paradigm
(the component that produces the returned string / uses most of the budget); its other components are
mechanism tags. A paradigm outside the vocabulary gets the family "provisional:<normalised name>": kept
separate from every other provisional family, never merged with them, and never granted the
new-family budget (a rewording cannot buy protection). Limits: the classification is only as fine as
the paradigm list (every vote-guided descent is `local_search`); many genuinely different programs
share one signature, since a finite vocabulary defines finitely many signatures.

The vocabulary (autoresearch/vocab/<problem>_v1.json) is frozen during compared runs; a change is a new
version file, and the version is part of every cache key and signature.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .json_cache import JsonCache, cache_key, text_hash
from .ledger import STATUS_NO_OUTPUT, STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD, Entry
from .llm_calls import claude_json
from .novelty import fingerprint

VOCAB_DIR = Path(__file__).resolve().parent / "vocab"
ROLES = ("paradigm", "mechanism", "detail")
_SPELLING = [("neighbor", "neighbour"), ("ization", "isation"), ("imize", "imise"), ("-", " "), ("_", " "), ("/", " ")]


def norm_term(text: str) -> str:
    """Lowercase, British spelling, hyphens/underscores as spaces, naive singular (moves -> move)."""
    t = text.lower()
    for a, b in _SPELLING:
        t = t.replace(a, b)
    words = []
    for w in t.split():
        if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


@dataclass(frozen=True)
class Term:
    id: str
    role: str
    definition: str
    synonyms: tuple[str, ...] = ()


class Vocab:
    def __init__(self, data: dict) -> None:
        self.version = data["version"]
        self.role_weights = {k: float(v) for k, v in data["role_weights"].items()}
        self.terms = {t["id"]: Term(t["id"], t["role"], t["definition"], tuple(t.get("synonyms", ())))
                      for t in data["terms"]}
        self.index: dict[str, dict[str, str]] = {r: {} for r in ROLES}
        for t in self.terms.values():
            for name in (t.id, *t.synonyms):
                self.index[t.role].setdefault(norm_term(name), t.id)
        self.hash = text_hash(json.dumps(data, sort_keys=True))

    @classmethod
    def load(cls, problem: str = "median_string", version: str = "v1") -> Vocab:
        base = problem.split("_long")[0]  # median_string_long shares the median_string vocabulary
        return cls(json.loads((VOCAB_DIR / f"{base}_{version}.json").read_text()))

    def ids(self, role: str) -> list[str]:
        return [t.id for t in self.terms.values() if t.role == role]

    def canonical(self, text: str, role: str | None = None) -> str | None:
        """Vocabulary id for a term or synonym (same role first, then any role); None if unknown."""
        key = norm_term(text)
        roles = [role] + [r for r in ROLES if r != role] if role else list(ROLES)
        for r in roles:
            if key in self.index[r]:
                return self.index[r][key]
        return None

    def role_of(self, term_id: str) -> str | None:
        t = self.terms.get(term_id)
        return t.role if t else None

    def prompt_block(self) -> str:
        out = []
        for role in ROLES:
            out.append(f"{role.upper()} ids:")
            out += [f"- {t.id}: {t.definition}" for t in self.terms.values() if t.role == role]
        return "\n".join(out)


@dataclass
class Signature:
    """A program's (or a proposal's) canonical description. Unknown terms are kept, flagged, low-weighted."""

    paradigm: str | None              # canonical paradigm id, None if outside the vocabulary
    paradigm_raw: str = ""            # the describer's wording when outside the vocabulary
    mechanisms: tuple[str, ...] = ()
    details: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()     # normalised terms the vocabulary does not cover
    summary: str = ""
    backend: str = "canonical"
    vocab_version: str = ""
    raw: dict = field(default_factory=dict)   # backend-specific payload (existing backend: Johann's descriptor)

    @property
    def family(self) -> str | None:
        if self.paradigm:
            return self.paradigm
        return f"provisional:{norm_term(self.paradigm_raw)}" if self.paradigm_raw.strip() else None

    @property
    def provisional(self) -> bool:
        return self.paradigm is None

    @property
    def empty(self) -> bool:
        return self.family is None and not self.mechanisms and not self.details and not self.unknown

    def weights(self, role_weights: dict[str, float]) -> dict[str, float]:
        """Term key -> role weight, for the weighted Jaccard distance (one key per term)."""
        w: dict[str, float] = {}
        if self.family:
            w[f"p:{self.family}"] = role_weights["paradigm"]
        for m in self.mechanisms:
            w[f"m:{m}"] = role_weights["mechanism"]
        for d in self.details:
            w[f"d:{d}"] = role_weights["detail"]
        for u in self.unknown:
            w.setdefault(f"u:{u}", role_weights.get("unknown", 1.0))
        return w

    def to_dict(self) -> dict:
        return {"paradigm": self.paradigm, "paradigm_raw": self.paradigm_raw, "mechanisms": list(self.mechanisms),
                "details": list(self.details), "unknown": list(self.unknown), "summary": self.summary,
                "backend": self.backend, "vocab_version": self.vocab_version, "raw": self.raw}

    @classmethod
    def from_dict(cls, d: dict) -> Signature:
        return cls(d.get("paradigm"), d.get("paradigm_raw", ""), tuple(d.get("mechanisms", ())),
                   tuple(d.get("details", ())), tuple(d.get("unknown", ())), d.get("summary", ""),
                   d.get("backend", "canonical"), d.get("vocab_version", ""), d.get("raw", {}) or {})

    def short(self) -> str:
        parts = [self.family or "?"] + list(self.mechanisms) + [f"({d})" for d in self.details]
        parts += [f"?{u}" for u in self.unknown]
        return " / ".join(parts)


# ----- signatures from structured LLM output -------------------------------------------------------

def signature_from_canonical(out: dict, vocab: Vocab) -> Signature:
    """Map the canonical describer's JSON to a Signature; anything outside the enums becomes unknown."""
    p = str(out.get("paradigm") or "")
    paradigm = p if vocab.role_of(p) == "paradigm" else None
    raw = "" if paradigm else str(out.get("paradigm_other") or ("" if p == "other" else p))
    if not paradigm and raw:  # a known paradigm written out in words still lands on its id
        cid = vocab.canonical(raw, "paradigm")
        if cid and vocab.role_of(cid) == "paradigm":
            paradigm, raw = cid, ""
    mech, det, unknown = [], [], []
    items = [(t, "mechanism") for t in out.get("mechanisms") or []] + [(t, "detail") for t in out.get("details") or []]
    if out.get("unknown_mechanism"):
        items.append((out["unknown_mechanism"], "mechanism"))
    for t, role in items:
        cid = t if vocab.role_of(str(t)) else vocab.canonical(str(t), role)
        r = vocab.role_of(cid) if cid else None
        if r == "mechanism":
            mech.append(cid)
        elif r == "detail":
            det.append(cid)
        else:  # outside the vocabulary (or a paradigm listed as a component): flagged, low weight
            unknown.append(norm_term(str(t)))
    dedup = lambda xs: tuple(dict.fromkeys(xs))  # noqa: E731
    return Signature(paradigm, raw, dedup(mech), dedup(det), dedup(x for x in unknown if x), " ".join(
        str(out.get("summary", "")).split())[:300], "canonical", vocab.version)


def signature_from_descriptor(d, vocab: Vocab) -> Signature:
    """Johann's tiered descriptor (existing backend) mapped onto the canonical ids where a synonym matches."""
    mech, det, unknown = [], [], []
    for term, tier in d.terms:
        if tier == 6:
            continue
        role = "mechanism" if tier == 3 else "detail"
        cid = vocab.canonical(term, role)
        if cid and vocab.role_of(cid) == "mechanism":
            mech.append(cid)
        elif cid and vocab.role_of(cid) == "detail":
            det.append(cid)
        else:
            unknown.append(norm_term(term))
    paradigm = vocab.canonical(d.core, "paradigm")
    paradigm = paradigm if paradigm and vocab.role_of(paradigm) == "paradigm" else None
    return Signature(paradigm, "" if paradigm else d.core, tuple(dict.fromkeys(mech)), tuple(dict.fromkeys(det)),
                     tuple(dict.fromkeys(unknown)), d.summary, "existing", vocab.version, raw=d.to_dict())


# ----- the canonical describer ------------------------------------------------------------------------

SYSTEM = "You classify optimisation programs with a fixed vocabulary of ids. Output only the requested JSON."

PROMPT = """Classify the algorithm the program below ACTUALLY IMPLEMENTS, using only the vocabulary ids.

Problem: {problem}

Vocabulary ({version}):
{vocab}

Rules:
- Read the code, not its comments, names or docstrings: list a mechanism only if the code executes it.
  Ignore boilerplate every solver shares (CPU-deadline check, returning the best string, calling the metric).
- paradigm: exactly one id, the component that produces the returned string or uses most of the budget.
  A hybrid gets its principal paradigm here and its other components as mechanisms. Use "other" only if no
  paradigm fits, and then name it in paradigm_other (a short generic noun phrase).
- mechanisms: the structural components the code implements (at most 8). details: incidental choices (at most 5).
- unknown_mechanism: only if the code implements an important mechanism absent from the list; give a short
  generic name and, in unknown_justification, the code construct that implements it. Otherwise leave it empty.
- summary: at most 25 words of compact pseudocode of what the code does.

Program:
```python
{source}
```"""


def canonical_schema(vocab: Vocab) -> dict:
    return {
        "type": "object",
        "properties": {
            "paradigm": {"type": "string", "enum": vocab.ids("paradigm") + ["other"]},
            "paradigm_other": {"type": "string"},
            "mechanisms": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": vocab.ids("mechanism")}},
            "details": {"type": "array", "maxItems": 5, "items": {"type": "string", "enum": vocab.ids("detail")}},
            "unknown_mechanism": {"type": "string"},
            "unknown_justification": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["paradigm", "mechanisms", "details", "summary"],
    }


class FamilyDescriber:
    """Describes candidate sources as Signatures (canonical or existing backend), cached on disk.

    Cache keys cover the code fingerprint, backend, model, prompt, schema, problem text and the vocabulary
    version/hash (canonical backend; the existing backend's vocabulary grows and is not keyed)."""

    def __init__(self, root: Path, problem: str, vocab: Vocab, backend: str = "canonical", model: str = "sonnet",
                 budget=None, kind: str = "describe", workers: int = 16) -> None:
        self.root, self.problem, self.vocab, self.backend, self.model = root, problem, vocab, backend, model
        self.budget, self.kind, self.workers = budget, kind, workers
        self.cache = JsonCache(root / f"signatures-{backend}.json")
        self.usage_path = root / "llm_usage.jsonl"
        self.hits = 0
        self._existing_vocab = None

    def key(self, source: str) -> str:
        if self.backend == "canonical":
            return cache_key(fp=fingerprint(source), backend="canonical-v1", model=self.model, prompt=text_hash(PROMPT),
                             system=text_hash(SYSTEM), schema=text_hash(json.dumps(canonical_schema(self.vocab))),
                             problem=text_hash(self.problem), vocab=f"{self.vocab.version}:{self.vocab.hash}")
        from .descriptors import describe_key
        return cache_key(existing=describe_key(source, self.problem, self.model), vocab_map=self.vocab.hash)

    def describe_one(self, source: str) -> Signature | None:
        k = self.key(source)
        hit = self.cache.get(k)
        if hit is not None:
            self.hits += 1
            return Signature.from_dict(hit)
        sig = self._call(source)
        if sig is not None:
            self.cache.put(k, sig.to_dict())
        return sig

    def _call(self, source: str) -> Signature | None:
        if self.backend == "existing":
            from .descriptors import Vocabulary, describe
            if self._existing_vocab is None:
                self._existing_vocab = Vocabulary.load(self.root / "vocab.json")
            try:
                d = describe(source, self._existing_vocab, self.problem, model=self.model, kind=self.kind,
                             cache_path=self.root / "descriptors.json", usage_path=self.usage_path, budget=self.budget)
            except RuntimeError:
                return None
            self._existing_vocab.add(d)
            return signature_from_descriptor(d, self.vocab)
        prompt = PROMPT.format(problem=self.problem, version=self.vocab.version, vocab=self.vocab.prompt_block(),
                               source=source)
        for attempt in (1, 2):
            out, rec = claude_json(prompt, canonical_schema(self.vocab), SYSTEM, self.model, kind=self.kind,
                                   usage_path=self.usage_path, attempt=attempt, budget=self.budget)
            if out:
                sig = signature_from_canonical(out, self.vocab)
                if not sig.empty:
                    return sig
            if rec.get("error") == "budget exhausted":
                return None
        return None

    def describe_many(self, sources: list[str]) -> list[Signature | None]:
        if not sources:
            return []
        with ThreadPoolExecutor(min(self.workers, len(sources))) as pool:
            return list(pool.map(self.describe_one, sources))


# ----- lineage, family archive, entropy -------------------------------------------------------------

def lineage_of(entry: Entry, by_id: dict[int, Entry]) -> int:
    """Founder id: walk first parents until the seed or a new_family proposal (which founds a lineage)."""
    seen: set[int] = set()
    e = entry
    while e.mode not in ("seed", "new_family") and e.parent_ids and e.parent_ids[0] in by_id and e.id not in seen:
        seen.add(e.id)
        e = by_id[e.parent_ids[0]]
    return e.id


def entry_signature(e: Entry, extra: dict[int, dict] | None = None) -> Signature | None:
    """The entry's signature: from its usage, else from the side table of entries described after they were
    recorded (the seed; the ledger is append-only)."""
    d = e.usage.get("signature") or (extra or {}).get(e.id, {}).get("signature")
    return Signature.from_dict(d) if d else None


def produced_family(e: Entry, by_id: dict[int, Entry], extra: dict[int, dict] | None = None) -> str | None:
    """Family the entry actually belongs to: its own signature; an exact duplicate inherits the family of the
    program it duplicates; sessions with no code (or blocked by the guard) have none."""
    if e.usage.get("family"):
        return e.usage["family"]
    if extra and extra.get(e.id, {}).get("family"):
        return extra[e.id]["family"]
    if e.status == STATUS_REJECTED_DUPLICATE and e.novelty.get("nearest_id") in by_id:
        src = by_id[e.novelty["nearest_id"]]
        return produced_family(src, by_id, extra) if src.id != e.id else None
    return None


@dataclass
class FamilyStats:
    family: str
    provisional: bool
    first_id: int
    first_generation: int | None
    entries: list[int] = field(default_factory=list)       # every try attributed to the family (incl. rejected)
    valid: list[int] = field(default_factory=list)          # scored entries
    best_id: int | None = None
    best_objective: float = math.inf
    records: list[tuple[int, float]] = field(default_factory=list)   # (entry id, objective) at each new record
    last_generation: int | None = None
    trials_since_record: int = 0


class FamilyArchive:
    """Per-family history derived from the ledger. Representatives never disappear when a family leaves the
    top-K (they become dormant); exact duplicates are recognised by the code gate over all candidates."""

    def __init__(self, entries: list[Entry], sense: str = "min", extra: dict[int, dict] | None = None) -> None:
        self.entries = entries
        self.by_id = {e.id: e for e in entries}
        self.sense, self.extra = sense, extra or {}
        self.families: dict[str, FamilyStats] = {}
        for e in entries:
            fam = produced_family(e, self.by_id, self.extra)
            if not fam:
                continue
            st = self.families.get(fam)
            if st is None:
                st = self.families[fam] = FamilyStats(fam, fam.startswith("provisional:"), e.id, e.generation)
            st.entries.append(e.id)
            st.last_generation = e.generation
            if e.scored and e.status not in (STATUS_REJECTED_DUPLICATE,):
                st.valid.append(e.id)
                if st.best_id is None or better(e.objective, st.best_objective, sense):
                    st.best_objective, st.best_id = e.objective, e.id
                    st.records.append((e.id, e.objective))
                    st.trials_since_record = 0
                    continue
            st.trials_since_record += 1

    def representatives(self) -> list[tuple[str, Entry]]:
        """Best scored program of every family, best first (dormant families included)."""
        reps = [(f, self.by_id[s.best_id]) for f, s in self.families.items() if s.best_id is not None]
        return sorted(reps, key=lambda fe: (fe[1].objective if self.sense == "min" else -fe[1].objective, fe[1].id))

    def family_of(self, e: Entry) -> str | None:
        return produced_family(e, self.by_id, self.extra)

    def signature_of(self, e: Entry) -> Signature | None:
        return entry_signature(e, self.extra)


def better(new: float, old: float, sense: str = "min") -> bool:
    return new < old if sense == "min" else new > old


def record_gain(new: float, best_before: float | None, sense: str = "min") -> float:
    """gain_f = max(0, improvement over the family record before this try), in objective units (no division
    by the previous score, so zero or negative scores are fine). No record yet -> 0."""
    if best_before is None or math.isinf(best_before) or new is None or math.isinf(new):
        return 0.0
    return max(0.0, best_before - new) if sense == "min" else max(0.0, new - best_before)


@dataclass
class EntropyStat:
    counts: dict[str, int]
    n: int
    families: int
    H: float                 # natural log; 0 for an empty set (by convention, see `empty`)
    effective: float         # exp(H): effective number of families
    normalized: float | None  # H / log(denominator), None when the denominator is <= 1
    denominator: int
    empty: bool

    def to_dict(self) -> dict:
        return {"counts": self.counts, "n": self.n, "families": self.families, "H": round(self.H, 4),
                "effective": round(self.effective, 3), "normalized": None if self.normalized is None else round(
                    self.normalized, 4), "denominator": self.denominator, "empty": self.empty}


def entropy(counts: dict[str, int], denominator: int) -> EntropyStat:
    """Shannon entropy H = -sum p log p of a family count table. Empty table: H = 0, empty = True.
    normalized = H / log(denominator) when denominator > 1 (e.g. min(K, number of vocabulary paradigms))."""
    counts = {k: v for k, v in counts.items() if v > 0}
    n = sum(counts.values())
    if n == 0:
        return EntropyStat({}, 0, 0, 0.0, 1.0, None, denominator, True)
    H = -sum((c / n) * math.log(c / n) for c in counts.values())
    norm = H / math.log(denominator) if denominator > 1 else None
    return EntropyStat(dict(Counter(counts).most_common()), n, len(counts), H, math.exp(H), norm, denominator, False)


def top_k_entries(entries: list[Entry], k: int) -> list[Entry]:
    scored = [e for e in entries if e.scored and e.confirmed is not False and e.status != STATUS_REJECTED_DUPLICATE]
    return sorted(scored, key=lambda e: (e.objective, e.id))[:k]


def top_k_entropy(archive: FamilyArchive, k: int, n_paradigms: int) -> EntropyStat:
    """Concentration of the best solutions: family counts among the top-k programs (unknown family = 'unknown')."""
    top = top_k_entries(archive.entries, k)
    counts = Counter(archive.family_of(e) or "unknown" for e in top)
    return entropy(dict(counts), min(k, n_paradigms))


def recent_entropy(archive: FamilyArchive, window: int, n_paradigms: int) -> EntropyStat:
    """Search effort: produced family of the last `window` finished tries (rejected and empty ones included,
    those without a family as 'unknown'; the seed excluded)."""
    tries = [e for e in archive.entries if e.mode != "seed"][-window:]
    counts = Counter(archive.family_of(e) or ("unknown" if e.status not in (STATUS_NO_OUTPUT, STATUS_REJECTED_GUARD)
                                                else "no_code") for e in tries)
    return entropy(dict(counts), min(window, n_paradigms))


def family_landscape(archive: FamilyArchive, limit: int = 14) -> str:
    lines = []
    for fam, e in archive.representatives()[:limit]:
        sig = archive.signature_of(e)
        st = archive.families[fam]
        lines.append(f"- {fam}: best #{e.id} objective={e.objective:g} ({len(st.entries)} tries)"
                     + (f" — {sig.short()}" if sig else ""))
    return "\n".join(lines) or "  (no family described yet)"


def json_safe(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=str))
