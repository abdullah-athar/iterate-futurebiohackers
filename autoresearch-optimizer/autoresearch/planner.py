"""Plan before code (semantic_retry_before_codegen): a short structured plan per task, checked and reserved
before its agent session starts, so a task that would repeat known work is redirected for the price of one
plan call instead of a full session.

    plan -> check -> reserve -> agent session (the plan is its direction) -> describe the code -> log plan/code gap

check, per mode (convention: too close <=> distance < threshold):
  new_family  distance to family representatives (archive_threshold) and to the plans already reserved in
              the batch (batch_threshold);
  tune        a concrete change is required; the same (parent, change kind, change target) as another plan
              of the batch is a repeat, identical tags to the parent are fine;
  merge       the same parent pair and combination target as another plan of the batch is a repeat;
  fix_losers  the same targeted failure as another plan of the batch is a repeat.
Too close -> regenerate with the neighbour, the shared terms and the dimension to change, keeping the mode
(at most max_proposal_regenerations); still too close -> the task is handed back to the scheduler, which
drops it (no silent switch to exploration) and the agent is not run. The plan's signature is provisional:
the description of the code that was actually written is the one the archive keeps.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from .distance import nearest
from .families import FamilyArchive, Signature, family_landscape, norm_term, signature_from_canonical
from .llm_calls import claude_json
from .prompts import MODES

CHANGE_KINDS = ["parameter", "mechanism", "structure", "paradigm", "bugfix", "combination"]

SYSTEM = "You plan one experiment for an optimisation-program search. Output only the requested JSON."

PROMPT = """Plan ONE experiment for the problem below, in mode `{mode}`, before any code is written.

Problem: {problem}

Mode instructions: {mode_text}

Parent program(s):
{parents}

Known families (best program of each):
{landscape}

Plans already reserved by other agents in this batch (do not repeat them):
{reserved}
{feedback}
Describe the program your experiment will produce with the vocabulary ids below, and the concrete change:
change.kind (parameter | mechanism | structure | paradigm | bugfix | combination), change.target (a vocabulary
id, a parameter name or the targeted failure), change.description (one sentence). summary: at most 25 words.

Vocabulary ({version}):
{vocab}"""


def plan_schema(vocab) -> dict:
    return {
        "type": "object",
        "properties": {
            "paradigm": {"type": "string", "enum": vocab.ids("paradigm") + ["other"]},
            "paradigm_other": {"type": "string"},
            "mechanisms": {"type": "array", "maxItems": 8, "items": {"type": "string", "enum": vocab.ids("mechanism")}},
            "details": {"type": "array", "maxItems": 5, "items": {"type": "string", "enum": vocab.ids("detail")}},
            "change": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": CHANGE_KINDS}, "target": {"type": "string"},
                "description": {"type": "string"}}, "required": ["kind", "target", "description"]},
            "summary": {"type": "string"},
        },
        "required": ["paradigm", "mechanisms", "details", "change", "summary"],
    }


class Planner:
    def __init__(self, layer) -> None:
        self.layer = layer
        self._lock = threading.Lock()
        self._batch: list[tuple[str, Signature, dict, tuple]] = []   # (label, signature, change, key)

    def _key(self, a, sig: Signature, change: dict) -> tuple:
        return (a.mode, tuple(sorted(a.parent_ids)) if a.mode == "merge" else tuple(a.parent_ids),
                change.get("kind"), norm_term(str(change.get("target", ""))))

    def check(self, a, sig: Signature, change: dict, refs, ref_ids) -> tuple[bool, str, dict]:
        """(ok, feedback for a regeneration, info). Batch comparisons see every plan reserved so far."""
        cfg, metric = self.layer.cfg, self.layer.metric
        if not str(change.get("description", "")).strip() or not str(change.get("target", "")).strip():
            return False, "The plan has no concrete change: name the target and describe the change.", {"why": "no change"}
        key = self._key(a, sig, change)
        with self._lock:
            batch = list(self._batch)
        for label, other, ochange, okey in batch:
            if okey == key:
                return False, (f"{label} already reserved the same change ({change.get('kind')} on "
                               f"{change.get('target')}) of the same parent(s). Change a different dimension: "
                               "another mechanism, another parameter or another target."), {"why": "same change", "neighbour": label}
        if a.mode == "new_family":
            pool = refs + [(label, s) for label, s, _, _ in batch]
            res, rid = nearest(sig, pool, metric)
            thr = (cfg.archive_threshold if rid in ref_ids else cfg.batch_threshold) if res.available else None
            info = {**res.to_dict(), "nearest_id": rid, "threshold": thr}
            if res.available and thr is not None and res.value < thr:
                shared = ", ".join(res.components.get("shared", [])) or "the paradigm"
                return False, (f"The plan is too close to {('#' + str(rid)) if rid in ref_ids else rid} (distance "
                               f"{res.value:.2f} < {thr}); it shares {shared}. Change the paradigm or the core "
                               "mechanisms, not the wording."), info
            return True, "", info
        return True, "", {"why": "ok"}

    def reserve(self, a, sig: Signature, change: dict) -> bool:
        key = self._key(a, sig, change)
        with self._lock:
            if any(k == key for _, _, _, k in self._batch):
                return False
            if self.layer.reservations.reserve(key, a.worker) is not None:
                return False
            self._batch.append((f"worker {a.worker}", sig, change, key))
            return True

    def plan_one(self, a, archive: FamilyArchive, refs, ref_ids) -> tuple[str, int, dict]:
        """Returns (status: accepted | dropped | unavailable, attempts, plan dict)."""
        layer, cfg = self.layer, self.layer.cfg
        parents = []
        for pid in a.parent_ids:
            e = archive.by_id.get(pid)
            s = archive.signature_of(e) if e else None
            parents.append(f"- #{pid} objective={e.objective:g}: {s.short() if s else '(not described)'}"
                           f"{' — ' + s.summary if s and s.summary else ''}" if e else f"- #{pid}")
        feedback, last = "", {}
        for attempt in range(1, cfg.max_proposal_regenerations + 2):
            with self._lock:
                reserved = "\n".join(f"- {label}: {s.short()} — change {c.get('kind')} on {c.get('target')}"
                                     for label, s, c, _ in self._batch) or "  (none yet)"
            prompt = PROMPT.format(mode=a.mode, problem=layer.describer.problem, mode_text=MODES[a.mode],
                                   parents="\n".join(parents) or "  (none)", landscape=family_landscape(archive),
                                   reserved=reserved, feedback=f"\nYour previous plan was refused: {feedback}\n" if feedback else "",
                                   version=layer.vocab.version, vocab=layer.vocab.prompt_block())
            out, rec = claude_json(prompt, plan_schema(layer.vocab), SYSTEM, cfg.describe_model, kind="plan",
                                   usage_path=layer.describer.usage_path, attempt=attempt, budget=layer.budget)
            if not out:
                return "unavailable", attempt, {"error": rec.get("error")}
            sig = signature_from_canonical(out, layer.vocab)
            change = out.get("change") or {}
            ok, feedback, info = self.check(a, sig, change, refs, ref_ids)
            last = {"signature": sig.to_dict(), "change": change, "summary": out.get("summary", ""), "check": info}
            if ok and self.reserve(a, sig, change):
                return "accepted", attempt, last
            if ok:
                feedback = "another agent reserved the same plan at the same moment; plan a different change."
        return "dropped", cfg.max_proposal_regenerations + 1, last

    def plan_all(self, assignments: list, gen: int, archive: FamilyArchive, emit, log: dict) -> list:
        with self._lock:
            self._batch.clear()
        refs = self.layer.family_refs(archive)
        ref_ids = {rid for rid, _ in refs}
        with ThreadPoolExecutor(max(1, len(assignments))) as pool:
            outcomes = list(pool.map(lambda a: self.plan_one(a, archive, refs, ref_ids), assignments))
        kept = []
        for a, (status, attempts, plan) in zip(assignments, outcomes):
            a.meta["plan_status"], a.meta["plan_attempts"] = status, attempts
            if emit:
                emit("plan", gen=gen, worker=a.worker, mode=a.mode, status=status, attempts=attempts,
                     summary=plan.get("summary", ""), change=plan.get("change"), check=plan.get("check"))
            if status == "dropped":
                continue  # handed back: the scheduler does not run this agent (its budget is saved)
            if status == "accepted":
                a.meta["plan_signature"], a.meta["plan_change"] = plan["signature"], plan["change"]
                a.direction = f"implement the reserved plan: {plan['summary']}"
                c = plan["change"]
                a.notes = (a.notes + "\n\n" if a.notes else "") + (
                    f"## Your plan (reserved for you in this batch)\n{plan['summary']}\nChange: {c.get('kind')} on "
                    f"{c.get('target')} — {c.get('description')}\nImplement this plan; if it proves unworkable, keep "
                    "the same mode and make the closest workable change.")
            kept.append(a)
        log["plans"] = {"accepted": sum(1 for s, _, _ in outcomes if s == "accepted"),
                        "dropped": sum(1 for s, _, _ in outcomes if s == "dropped"),
                        "unavailable": sum(1 for s, _, _ in outcomes if s == "unavailable"),
                        "regenerations": sum(max(0, n - 1) for _, n, _ in outcomes)}
        return kept
