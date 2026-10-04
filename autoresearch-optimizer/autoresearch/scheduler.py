"""Diversity scheduler: composes the existing mode bandit with family-level policies (all off by default).

The UCB bandit's arms are the four prompt modes; that does not change. A "batch" is one swarm generation:
its assignments are planned together before any agent starts, so "the same step" means the same
generation. Per batch, in this order, each layer only rewrites slots the bandit already allocated (it moves
budget, it never adds agents) and records why:

  1. family_grace       a new admissible family gets funded refinements: its best program is refined in
                        `tune` mode (minimum 2 evaluations, +1 per record gain > epsilon, at most 6).
  2. entropy_controller when the top-k is concentrated on few families AND the global best stagnates,
                        some slots refine under-explored families or ask for a new approach.
  3. diverse_parents    remaining tune/merge slots get parents that are good *and* different.
  4. focus/reservation  two slots on the same parent get different mechanisms to work on; planned work is
                        reserved atomically so two tasks of a batch never launch the same plan.
(1) and (2) together change at most `max_override_fraction` of the slots. Each slot keeps `ucb_mode`
(the bandit's decision); an `override` field says what changed it and why.

After the agents, `after_precheck` describes the code (the code's description is authoritative for the
archive), assigns families and applies the distance policy per mode:
  new_family  minimum distance to family representatives and to the batch's accepted proposals
              (observe: log; soft: log + the agents saw the family landscape; strict: drop if closer
              than a *calibrated* archive_threshold);
  tune        closeness to the parent is expected; a large distance is logged as a mode drift, never a
              rejection (an identical-tag refinement is fine);
  merge       flagged if the same parent pair already produced a near-identical combination;
  fix_losers  never judged on descriptors (a bug fix can leave the signature unchanged).
Convention: too close  <=>  minimum distance < threshold (raising a threshold demands more difference).
"""

from __future__ import annotations

import math
import random
import threading
from collections import Counter
from dataclasses import asdict, dataclass

from .distance import get_metric, nearest
from .families import (
    FamilyArchive,
    FamilyDescriber,
    Signature,
    Vocab,
    family_landscape,
    lineage_of,
    recent_entropy,
    record_gain,
    top_k_entropy,
)
from .json_cache import JsonCache
from .ledger import STATUS_REJECTED_DUPLICATE, Entry
from .novelty import NoveltyVerdict
from .prompts import MODES

POLICIES = ("off", "observe", "soft", "strict")


def diversity_enabled(cfg) -> bool:
    return (cfg.distance_policy != "off" or cfg.diverse_parent_selection or cfg.semantic_retry_before_codegen
            or cfg.entropy_controller or cfg.family_grace)


def needs_families(cfg) -> bool:
    return diversity_enabled(cfg) or cfg.family_diagnostics


def validate_config(cfg) -> list[str]:
    errs = []
    if cfg.distance_policy not in POLICIES:
        errs.append(f"distance_policy must be one of {POLICIES}")
    if cfg.descriptor_backend not in ("canonical", "existing"):
        errs.append("descriptor_backend must be canonical or existing")
    if cfg.distance_metric == "minilm" and cfg.descriptor_backend != "existing":
        errs.append("the minilm distance needs descriptor_backend=existing")
    if cfg.distance_policy == "strict" and cfg.archive_threshold is None:
        errs.append("distance_policy=strict needs a calibrated --archive-threshold (scripts/calibrate_distance.py)")
    if cfg.semantic_retry_before_codegen and cfg.batch_threshold is None and cfg.archive_threshold is None:
        errs.append("semantic_retry_before_codegen needs --batch-threshold or --archive-threshold")
    if cfg.descriptors and needs_families(cfg):
        errs.append("Johann's layer (descriptors) and the diversity scheduler are separate arms: use --no-descriptors")
    if not 1 <= cfg.grace_min_evaluations <= cfg.grace_max_evaluations:
        errs.append("need 1 <= grace_min_evaluations <= grace_max_evaluations")
    return errs


# ----- reservations ------------------------------------------------------------------------------------

class Reservations:
    """Check-and-insert of planned work under one lock: the first task to reserve a key owns it; another
    task asking for the same key gets the owner's id back and must plan something else. Released when a
    task fails or gives up; cleared at the next batch."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[tuple, int] = {}

    def reserve(self, key: tuple, worker: int) -> int | None:
        with self._lock:
            owner = self._held.get(key)
            if owner is None or owner == worker:
                self._held[key] = worker
                return None
            return owner

    def release(self, key: tuple, worker: int) -> None:
        with self._lock:
            if self._held.get(key) == worker:
                del self._held[key]

    def clear(self) -> None:
        with self._lock:
            self._held.clear()

    def held(self) -> dict[tuple, int]:
        with self._lock:
            return dict(self._held)


# ----- family grace (minimal budget for a new family) -------------------------------------------------

@dataclass
class Protection:
    family: str
    status: str          # active | queued | expired | refused
    first_id: int        # first valid (scored) program of the family
    used: int            # funded evaluations: the first program + every grace try, whatever its outcome
    budget: int          # min(grace_max, grace_min + improvements + patience)
    improvements: int    # family records that beat the previous record by more than epsilon
    reason: str = ""


def grace_states(archive: FamilyArchive, initial_families: set[str], cfg, can_fund: bool) -> list[Protection]:
    """Protection of every family discovered after the start, in discovery order (oldest obligation first).

    A family is admissible once it has a valid program, is not provisional (outside the vocabulary) and was
    not known at the start. Every grace try counts against its budget, so a family whose candidates keep
    failing loses its protection; identity is the canonical paradigm id, so renaming, a new lineage or a
    return from dormancy never renews it. Expiry only removes the bonus: the family stays in the archive."""
    grace_ids: dict[str, set[int]] = {}
    for e in archive.entries:
        if e.usage.get("grace_family"):
            grace_ids.setdefault(e.usage["grace_family"], set()).add(e.id)
    out: list[Protection] = []
    active = 0
    for st in sorted((s for s in archive.families.values() if s.valid), key=lambda s: s.valid[0]):
        if st.family in initial_families:
            continue
        if st.provisional:
            out.append(Protection(st.family, "refused", st.valid[0], 0, 0, 0, "provisional family (outside the vocabulary)"))
            continue
        used, improvements, expired = _replay_grace(st, grace_ids.get(st.family, set()), archive, cfg)
        budget = min(cfg.grace_max_evaluations, cfg.grace_min_evaluations + improvements + cfg.grace_patience)
        p = Protection(st.family, "active", st.valid[0], used, budget, improvements)
        if expired:
            p.status, p.reason = "expired", f"used {used}/{budget} funded evaluations"
        elif active >= cfg.max_protected_families:
            p.status, p.reason = "queued", f"{cfg.max_protected_families} families already protected"
        elif not can_fund and used == 1:
            p.status, p.reason = "refused", "no budget left for the minimal refinement"
        else:
            active += 1
        out.append(p)
    return out


def _replay_grace(st, grace_ids: set[int], archive: FamilyArchive, cfg) -> tuple[int, int, bool]:
    """Replay the family's tries in id order from its first valid program: every evaluated grace try uses one
    funded evaluation (also when it failed or drifted to another family); a try refused as a code duplicate
    was never evaluated, so it does not use the budget, but every grace try counts against grace_max_attempts
    so a family whose agents keep returning copies still expires. Every record beating the previous one by
    more than epsilon adds one to the budget. Stops at expiry, so expiry is final: a later record (made by a
    normal try) does not bring the bonus back. Returns (used, improvements, expired)."""
    first = st.valid[0]
    record_ids = {eid for eid, _ in st.records}
    prev = archive.by_id[first].objective
    used, improvements, attempts = 1, 0, 0
    budget = min(cfg.grace_max_evaluations, cfg.grace_min_evaluations + cfg.grace_patience)
    if used >= budget:
        return used, improvements, True
    for eid in sorted((set(st.entries) | grace_ids) - {first}):
        if eid < first:
            continue
        e = archive.by_id[eid]
        if eid in grace_ids:
            attempts += 1
            if e.status != STATUS_REJECTED_DUPLICATE:
                used += 1
            if attempts >= cfg.grace_max_attempts:
                return used, improvements, True
        if eid in record_ids:
            if record_gain(e.objective, prev, archive.sense) > cfg.grace_epsilon:
                improvements += 1
            prev = e.objective
        budget = min(cfg.grace_max_evaluations, cfg.grace_min_evaluations + improvements + cfg.grace_patience)
        if used >= budget:
            return used, improvements, True
    return used, improvements, False


# ----- concentration + stagnation controller ----------------------------------------------------------

class EntropyController:
    """Turns exploration up while the top-k is concentrated AND the global best stagnates; turns it back
    down on a new global best or once the top-k spreads (hysteresis), updated once per batch."""

    def __init__(self, cfg, n_paradigms: int) -> None:
        self.cfg, self.n_paradigms = cfg, n_paradigms
        self.active = False

    def update(self, archive: FamilyArchive, generations_since_improvement: int) -> dict:
        cfg = self.cfg
        top = top_k_entropy(archive, cfg.top_k, self.n_paradigms)
        recent = recent_entropy(archive, cfg.allocation_window, self.n_paradigms)
        stagnating = generations_since_improvement >= cfg.stagnation_generations
        single = top.n >= 2 and top.families == 1
        concentrated = single or (top.normalized is not None and top.normalized <= cfg.concentration_threshold)
        was = self.active
        if not was and concentrated and stagnating:
            self.active, why = True, "concentrated top-k and stagnating best"
        elif was and (not stagnating or (top.normalized is not None and top.normalized > cfg.concentration_release)):
            self.active, why = False, "new global best" if not stagnating else "top-k spread out"
        else:
            why = "unchanged"
        return {"top": top.to_dict(), "recent": recent.to_dict(), "stagnating": stagnating,
                "generations_since_improvement": generations_since_improvement, "concentrated": concentrated,
                "active": self.active, "changed": self.active != was, "why": why}


def under_explored(archive: FamilyArchive, recent: dict, dominant: str | None, cfg) -> list[str]:
    """Known, non-provisional families other than the dominant one, least recent effort first; a family
    already tried a lot without progress is not a priority just because it is absent from the top."""
    share = recent.get("counts", {})
    out = []
    for fam, st in archive.families.items():
        if fam == dominant or st.provisional or st.best_id is None:
            continue
        if len(st.entries) >= cfg.family_stagnation_trials and st.trials_since_record >= cfg.family_stagnation_trials:
            continue
        out.append((share.get(fam, 0), st.best_objective, fam))
    return [f for _, _, f in sorted(out)]


# ----- the layer -----------------------------------------------------------------------------------------

class DiversityLayer:
    """Hooks called by swarm.run_swarm: `plan` after the bandit's allocation, `after_precheck` after the code
    gate (describe + distance policy), `after_generation` for the per-batch family/entropy log."""

    def __init__(self, run, budget=None) -> None:
        cfg = run.config
        errs = validate_config(cfg)
        if errs:
            raise ValueError("; ".join(errs))
        self.run, self.cfg, self.budget = run, cfg, budget
        self.vocab = Vocab.load(run.problem_name)
        self.metric = get_metric(cfg.distance_metric, self.vocab.role_weights)
        problem_text = " ".join(run.problem.describe().strip().splitlines()[:4])
        kind = "describe" if diversity_enabled(cfg) else "describe:diagnostic"
        self.describer = FamilyDescriber(run.store.root, problem_text, self.vocab, cfg.descriptor_backend,
                                         cfg.describe_model, budget, kind)
        self.side = JsonCache(run.store.root / "entry_signatures.json")
        self.controller = EntropyController(cfg, len(self.vocab.ids("paradigm")))
        self.reservations = Reservations()
        self.planner = None
        if cfg.semantic_retry_before_codegen:
            from .planner import Planner
            self.planner = Planner(self)
        self._ensure_described([e for e in run.entries() if e.mode == "seed"])

    # ----- family bookkeeping ---------------------------------------------------------------------
    def extra(self) -> dict[int, dict]:
        path = self.side.path
        if not path.exists():
            return {}
        import json
        return {int(k): v for k, v in json.loads(path.read_text()).items()}

    def archive(self, entries: list[Entry] | None = None) -> FamilyArchive:
        return FamilyArchive(entries if entries is not None else self.run.entries(), "min", self.extra())

    def _ensure_described(self, entries: list[Entry]) -> None:
        """Describe recorded programs that have no signature yet (the seed) into the side table."""
        extra = self.extra()
        todo = [e for e in entries if e.source_path and not e.usage.get("signature") and e.id not in extra]
        sigs = self.describer.describe_many([self.run.store.read_candidate(e) for e in todo])
        for e, sig in zip(todo, sigs):
            if sig is not None:
                self.side.put(str(e.id), {"signature": sig.to_dict(), "family": sig.family})

    def initial_families(self, archive: FamilyArchive) -> set[str]:
        return {f for e in archive.entries if e.mode == "seed" and (f := archive.family_of(e))}

    def family_refs(self, archive: FamilyArchive) -> list[tuple[int, Signature | None]]:
        """One representative per family (dormant ones included). Past `max_family_refs`, keep half by
        best objective and half by most recent activity, so it is not only the best-ranked that survive."""
        reps = archive.representatives()
        if len(reps) > self.cfg.max_family_refs:
            half = self.cfg.max_family_refs // 2
            recent = sorted(reps, key=lambda fe: -(archive.families[fe[0]].last_generation or 0))
            keep = {f for f, _ in reps[:half]} | {f for f, _ in recent[:self.cfg.max_family_refs - half]}
            reps = [fe for fe in reps if fe[0] in keep]
        return [(e.id, archive.signature_of(e)) for _, e in reps]

    def can_fund_refinement(self) -> bool:
        if self.budget is None or not self.budget.capped or self.budget.max_usd is None:
            return True
        from .swarm import _per_agent_estimate
        est = _per_agent_estimate(self.run, None) or 0.0
        return self.budget.spent().usd + est <= self.budget.max_usd

    # ----- planning -------------------------------------------------------------------------------------
    def plan(self, assignments: list, gen: int, rng: random.Random, emit=None) -> list:
        cfg, run = self.cfg, self.run
        entries = run.entries()
        archive = self.archive(entries)
        best = run.archive(entries).global_best
        self.reservations.clear()
        for a in assignments:
            a.meta.setdefault("ucb_mode", a.mode)
        cap = math.floor(cfg.max_override_fraction * len(assignments))
        overridden: set[int] = set()
        log: dict = {"gen": gen, "ucb_modes": [a.mode for a in assignments], "overrides": [], "cap": cap}

        if cfg.family_grace:
            prots = grace_states(archive, self.initial_families(archive), cfg, self.can_fund_refinement())
            log["protections"] = [asdict(p) for p in prots]
            for p in (p for p in prots if p.status == "active"):
                if len(overridden) >= cap:
                    log["overrides"].append({"family": p.family, "deferred": "override cap reached"})
                    break
                a = self._take_slot(assignments, overridden, prefer="tune")
                parent = archive.by_id[archive.families[p.family].best_id]
                why = ("grace:initial" if p.used == 1 else "grace:progress" if p.improvements else "grace:patience")
                self._override(a, "tune", [parent.id], why, target_family=p.family, grace_family=p.family,
                               grace_used=p.used, grace_budget=p.budget)
                a.direction = f"refine the best {p.family} program #{parent.id} without leaving its family"
                a.notes = (a.notes + "\n\n" if a.notes else "") + (
                    f"## Protected family: {p.family}\nThis family is new and gets a few funded refinements "
                    f"({p.used} of {p.budget} used). Improve program #{parent.id} while keeping its paradigm "
                    f"({p.family}); change one mechanism or its parameters, concretely.")
                overridden.add(a.worker)
                log["overrides"].append({"worker": a.worker, "from": a.meta["ucb_mode"], "to": "tune",
                                         "reason": why, "family": p.family, "parent": parent.id})

        if cfg.entropy_controller:
            state = self.controller.update(archive, run.generations_since_improvement(entries))
            log["controller"] = state
            if state["active"] and len(overridden) < cap:
                top_counts = state["top"]["counts"]
                dominant = max(top_counts, key=top_counts.get) if top_counts else None
                targets = under_explored(archive, state["recent"], dominant, cfg)
                untried = [p for p in self.vocab.ids("paradigm") if p not in archive.families]
                n = min(max(1, round(cfg.explore_boost_fraction * len(assignments))), cap - len(overridden))
                for i in range(n):
                    a = self._take_slot(assignments, overridden, prefer="tune")
                    if i % 2 == 0 and targets:
                        fam = targets.pop(0)
                        rep = archive.by_id[archive.families[fam].best_id]
                        self._override(a, "tune", [rep.id], "controller:under-explored family", target_family=fam)
                        a.direction = f"refine #{rep.id}, the best {fam} program (an under-explored family)"
                    else:
                        self._override(a, "new_family", [best.id], "controller:new approach", target_family=None)
                        a.direction = ("a new approach, outside the families that dominate the best solutions ("
                                       + ", ".join(list(top_counts)[:3]) + "); untried paradigms include: "
                                       + ", ".join(untried[:6]))
                    overridden.add(a.worker)
                    log["overrides"].append({"worker": a.worker, "from": a.meta["ucb_mode"], "to": a.mode,
                                             "reason": a.meta["override"], "family": a.meta.get("target_family")})

        if cfg.diverse_parent_selection:
            self._diverse_parents(assignments, overridden, archive, best, rng, log)

        self._focus_and_reserve(assignments, archive, log)

        if cfg.distance_policy in ("soft", "strict"):
            landscape = family_landscape(archive)
            for a in assignments:
                if a.mode == "new_family":
                    a.notes = (a.notes + "\n\n" if a.notes else "") + (
                        "## Known families (best program of each)\n" + landscape + "\nYour solver must differ "
                        "substantially from these in its paradigm or core mechanisms; rewording is not a difference.")
        if self.planner is not None:
            assignments = self.planner.plan_all(assignments, gen, archive, emit, log)
        log["final"] = [{"worker": a.worker, "mode": a.mode, "parents": a.parent_ids, **{
            k: v for k, v in a.meta.items() if k in ("ucb_mode", "override", "target_family", "focus", "parent_reason")}}
            for a in assignments]
        if emit:
            emit("schedule", **log)
        return assignments

    @staticmethod
    def _override(a, mode: str, parents: list[int], reason: str, **meta) -> None:
        a.mode, a.parent_ids = mode, parents
        a.meta["override"] = reason
        a.meta.update(meta)

    @staticmethod
    def _take_slot(assignments: list, overridden: set[int], prefer: str):
        free = [a for a in assignments if a.worker not in overridden]
        pref = [a for a in free if a.mode == prefer]
        if pref:
            return pref[-1]
        counts = Counter(a.mode for a in free)
        mode = max(counts, key=lambda m: (counts[m], -list(MODES).index(m)))
        return [a for a in free if a.mode == mode][-1]

    def _diverse_parents(self, assignments, overridden, archive: FamilyArchive, best: Entry, rng, log) -> None:
        """tune: greedy max-min over good programs (within parent_quality_margin of the best) using the
        signature distance; merge: the complementary member most distant from the best, untried pairs first."""
        good = {e.id: e for _, e in archive.representatives()}
        limit = best.objective * (1 + self.cfg.parent_quality_margin)
        cands = [e for e in good.values() if e.objective <= limit and e.id != best.id]
        picked = [best]
        while cands:
            def score(c):
                ds = [self.metric.distance(archive.signature_of(c), archive.signature_of(p)) for p in picked]
                return min((d.value if d.available else (0.0 if archive.family_of(c) == archive.family_of(p)
                                                          else 1.0)) for d, p in zip(ds, picked))
            c = max(cands, key=score)
            if score(c) <= 0.0:
                break
            picked.append(c)
            cands.remove(c)
        tunes = [a for a in assignments if a.mode == "tune" and a.worker not in overridden]
        for i, a in enumerate(tunes):
            p = picked[i % len(picked)]
            a.parent_ids = [p.id]
            a.meta["parent_reason"] = "best" if p.id == best.id else f"diverse: {archive.family_of(p)} representative"
        merges = [a for a in assignments if a.mode == "merge" and a.worker not in overridden]
        if merges:
            comp = [e for e, _ in self.run.archive().complementary()]
            tried = Counter(tuple(sorted(e.parent_ids)) for e in archive.entries if e.mode == "merge")
            bsig = archive.signature_of(best)

            def merge_key(e):
                d = self.metric.distance(archive.signature_of(e), bsig)
                return (tried[tuple(sorted((best.id, e.id)))], -(d.value if d.available else 0.0), e.id)
            order = sorted(comp, key=merge_key)
            for i, a in enumerate(merges):
                if order:
                    other = order[i % len(order)]
                    a.parent_ids = [best.id, other.id]
                    a.meta["parent_reason"] = f"diverse merge with {archive.family_of(other)} (tried {tried[tuple(sorted((best.id, other.id)))]}x)"
        log["diverse_parents"] = [p.id for p in picked]

    def _focus_and_reserve(self, assignments, archive: FamilyArchive, log) -> None:
        """Slots of one mode on the same parent(s) get different mechanisms of that parent to change, so two
        refinements of one program are two different experiments; each slot's (mode, parents, target family,
        focus) is reserved so the batch holds no duplicate task."""
        groups: dict[tuple, list] = {}
        for a in assignments:
            groups.setdefault((a.mode, tuple(a.parent_ids)), []).append(a)
        for (mode, parents), group in groups.items():
            # a prompt change, so it belongs to the distance/diversity policy (B), not to grace/controller (C)
            if not self.cfg.diverse_parent_selection or mode not in ("tune", "fix_losers") or len(group) < 2:
                continue
            sig = archive.signature_of(archive.by_id[parents[0]]) if parents and parents[0] in archive.by_id else None
            mechs = list(sig.mechanisms) if sig else []
            for i, a in enumerate(group):
                if mechs:
                    a.meta["focus"] = mechs[i] if i < len(mechs) else "a mechanism the parent does not have yet"
                    a.notes = (a.notes + "\n\n" if a.notes else "") + (
                        f"## Focus\nAnother agent refines the same parent. Make your change about: {a.meta['focus']}.")
        for a in assignments:
            key = (a.mode, tuple(a.parent_ids), a.meta.get("target_family"), a.meta.get("focus"), a.direction)
            owner = self.reservations.reserve(key, a.worker)
            if owner is not None:
                a.meta["focus"] = f"{a.meta.get('focus') or 'any'} (not the same as worker {owner})"
                self.reservations.reserve(key[:3] + (a.meta["focus"], a.direction), a.worker)
        log["reserved"] = len(self.reservations.held())

    # ----- after the code gate ---------------------------------------------------------------------------
    def after_precheck(self, results: list[dict], gen: int, emit=None) -> None:
        from .swarm import passed
        cfg = self.cfg
        todo = passed(results)
        sigs = self.describer.describe_many([r["source"] for r in todo])
        entries = self.run.entries()
        archive = self.archive(entries)
        by_id = archive.by_id
        refs = self.family_refs(archive) if cfg.distance_policy != "off" else []
        ref_ids = {rid for rid, _ in refs}
        batch: list[tuple[str, Signature | None]] = []
        merges_before = [e for e in entries if e.mode == "merge"]
        counts = Counter()
        for r, sig in zip(todo, sigs):
            a, u = r["assignment"], r["usage"]
            u["signature"] = sig.to_dict() if sig else None
            u["family"] = sig.family if sig else None
            u["family_provisional"] = bool(sig and sig.provisional)
            parent = by_id.get(a.parent_ids[0]) if a.parent_ids else None
            u["lineage"] = "new" if a.mode == "new_family" else (lineage_of(parent, by_id) if parent else None)
            if parent is not None:
                pf = archive.family_of(parent)
                u["parent_family"] = pf
                u["family_change"] = bool(sig and pf and sig.family != pf)
            if a.meta.get("plan_signature") and sig is not None:  # the code's description is authoritative
                gap = self.metric.distance(Signature.from_dict(a.meta["plan_signature"]), sig)
                u["plan_code_distance"] = gap.value
                if gap.available and gap.value > 0 and emit:
                    emit("plan_divergence", gen=gen, worker=a.worker, distance=round(gap.value, 4),
                         only_plan=gap.components.get("only_a"), only_code=gap.components.get("only_b"))
            if cfg.distance_policy == "off":
                continue
            decision, info = self._decide(a, sig, archive, refs, ref_ids, batch, merges_before)
            u["distance"] = info
            u["distance_decision"] = decision
            counts[decision] += 1
            if decision == "semantic_reject":
                nid = info.get("nearest_id")
                r["pre"] = (r["pre"][0], NoveltyVerdict(r["pre"][1].fingerprint, 1.0 - info["value"],
                                                        nid if isinstance(nid, int) else None, True))
                r["note"] = (f"semantic proximity: distance {info['value']:.3f} < {info['threshold']} to "
                             f"#{nid} ({info.get('metric')}); not evaluated")
                u["gate_dropped"] = True
                continue
            if decision == "mode_drift" and emit:
                emit("mode_drift", gen=gen, worker=a.worker, mode=a.mode, parent=a.parent_ids,
                     distance=info.get("value"), from_family=u.get("parent_family"), to_family=u.get("family"))
            batch.append((f"w{a.worker}", sig))
        if emit and cfg.distance_policy != "off":
            emit("distance_gate", gen=gen, policy=cfg.distance_policy, metric=self.metric.name,
                 version=self.metric.version, decisions=dict(counts), described=sum(1 for s in sigs if s),
                 undescribed=sum(1 for s in sigs if not s), cache_hits=self.describer.hits)

    def _decide(self, a, sig, archive, refs, ref_ids, batch, merges_before) -> tuple[str, dict]:
        cfg = self.cfg
        if sig is None:
            return "unavailable", {"available": False, "reason": "describe failed or budget exhausted"}
        if a.mode == "new_family":
            res, rid = nearest(sig, refs + batch, self.metric)
            thr = (cfg.archive_threshold if rid in ref_ids else cfg.batch_threshold) if res.available else None
            info = {**res.to_dict(), "nearest_id": rid, "threshold": thr,
                    "against": "archive" if rid in ref_ids else "batch"}
            if not res.available or thr is None:
                return "observed", info
            if res.value < thr:
                return ("semantic_reject" if cfg.distance_policy == "strict" else "proximity_flagged"), info
            return "accepted", info
        if a.mode == "tune":
            parent = archive.by_id.get(a.parent_ids[0]) if a.parent_ids else None
            res = self.metric.distance(sig, archive.signature_of(parent) if parent else None)
            info = {**res.to_dict(), "nearest_id": parent.id if parent else None, "threshold": cfg.drift_threshold_max,
                    "against": "parent"}
            if res.available and cfg.drift_threshold_max is not None and res.value > cfg.drift_threshold_max:
                return "mode_drift", info
            return "accepted", info
        if a.mode == "merge":
            pair = tuple(sorted(a.parent_ids))
            prior = [(e.id, archive.signature_of(e)) for e in merges_before if tuple(sorted(e.parent_ids)) == pair]
            res, rid = nearest(sig, prior, self.metric)
            info = {**res.to_dict(), "nearest_id": rid, "threshold": cfg.batch_threshold, "against": "same-pair merges"}
            if res.available and cfg.batch_threshold is not None and res.value < cfg.batch_threshold:
                return "combination_repeat", info
            return "accepted", info
        return "not_gated", {"available": False, "reason": f"{a.mode} is not judged on descriptors"}

    # ----- per-batch log -------------------------------------------------------------------------------
    def after_generation(self, gen: int, emit=None) -> dict:
        entries = self.run.entries()
        archive = self.archive(entries)
        n_par = len(self.vocab.ids("paradigm"))
        top = top_k_entropy(archive, self.cfg.top_k, n_par)
        recent = recent_entropy(archive, self.cfg.allocation_window, n_par)
        prots = grace_states(archive, self.initial_families(archive), self.cfg, True) if self.cfg.family_grace else []
        new = [f for f, st in archive.families.items() if any(archive.by_id[i].generation == gen for i in st.entries[:1])]
        rec = {"gen": gen, "top": top.to_dict(), "recent": recent.to_dict(), "families": len(archive.families),
               "new_families": new, "vocab": self.vocab.version, "metric": self.metric.version,
               "per_family": {f: {"tries": len(st.entries), "valid": len(st.valid), "best": st.best_objective,
                                  "best_id": st.best_id, "records": len(st.records),
                                  "since_record": st.trials_since_record, "provisional": st.provisional}
                              for f, st in archive.families.items()},
               "protections": [asdict(p) for p in prots]}
        if emit:
            emit("families", **rec)
        return rec
