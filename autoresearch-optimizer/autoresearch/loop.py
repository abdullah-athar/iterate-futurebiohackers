"""The research loop: propose -> novelty gate -> cascade evaluate -> archive -> ledger."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from .archive import Archive
from .guard import check_imports
from .ledger import (
    STATUS_EVALUATED,
    STATUS_FAILED,
    STATUS_KEPT,
    STATUS_REJECTED_DUPLICATE,
    STATUS_REJECTED_GUARD,
    STATUS_REJECTED_SCREEN,
    STATUS_SEED,
    VERDICT_FALSIFIED,
    VERDICT_INCONCLUSIVE,
    VERDICT_PARTIAL,
    VERDICT_SUPPORTED,
    VERDICT_UNCONFIRMED,
    VERDICT_UNTESTED,
    Entry,
    RunStore,
)
from .novelty import check_novelty
from .problem import EvalResult, Problem, get_problem
from .prompts import (
    MODES,
    Context,
    format_diagnostics,
    format_digest,
)
from .sandbox import evaluate_in_subprocess


@dataclass
class LoopConfig:
    problem: str = "median_string"
    novelty_threshold: float = 0.95
    max_novelty_attempts: int = 3
    exploit: float = 0.7
    patience: int = 4          # proposals without global improvement before 'plateau'
    ucb_c: float = 0.8

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class ResearchRun:
    def __init__(self, store: RunStore, problem: Problem | None = None) -> None:
        self.store = store
        cfg = store.config() if store.exists else {}
        self.config = LoopConfig(**{k: v for k, v in cfg.items() if k in LoopConfig.__dataclass_fields__})
        self.problem_name = self.config.problem
        self.problem = problem or get_problem(self.problem_name)

    # ----- lifecycle -------------------------------------------------------------------
    @classmethod
    def create(cls, store: RunStore, config: LoopConfig, seed_source: str | None = None) -> ResearchRun:
        if store.exists:
            raise FileExistsError(f"run already exists at {store.root}")
        store.create(config.to_dict())
        run = cls(store)
        source = seed_source or run.problem.seed_source()
        entry = Entry(id=0, parent_ids=[], mode="seed", hypothesis="Seed solver (starting point)", status=STATUS_SEED,
                      proposer="seed")
        entry.source_path = store.write_candidate(0, source)
        run._cascade(entry, source)
        if entry.scored:
            entry.improved_global = True
            entry.confirmed = run._confirm(entry, source)
        store.append(entry)
        return run

    def entries(self) -> list[Entry]:
        return self.store.entries()

    def archive(self, entries: list[Entry] | None = None) -> Archive:
        return Archive.build(entries if entries is not None else self.entries(), self.problem.objective_split)

    # ----- mode selection (UCB1 bandit over prompt modes, plateau-aware) ------------------
    def steps_since_improvement(self, entries: list[Entry]) -> int:
        n = 0
        for e in reversed(entries):
            if e.status == STATUS_SEED:
                break
            if e.improved_global:
                break
            if e.status not in (STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD):
                n += 1
        return n

    def choose_mode(self, entries: list[Entry], archive: Archive, rng: random.Random) -> str:
        eligible = [m for m in MODES if m != "merge" or archive.complementary()]
        plateau = self.steps_since_improvement(entries) >= self.config.patience
        if plateau:
            eligible = [m for m in eligible if m != "tune"] or eligible
        stats = {m: [0, 0.0] for m in MODES}
        for e in entries:
            if e.mode in stats and e.status not in (STATUS_SEED, STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD):
                stats[e.mode][0] += 1
                stats[e.mode][1] += 1.0 if e.improved_global else (0.5 if e.improved_instances else 0.0)
        total = sum(n for n, _ in stats.values()) or 1
        untried = [m for m in eligible if stats[m][0] == 0]
        if untried:
            return untried[0] if not plateau else rng.choice(untried)
        return max(eligible, key=lambda m: stats[m][1] / stats[m][0] + self.config.ucb_c * math.sqrt(math.log(total) / stats[m][0]))

    # ----- context for the proposer (shared by API mode and agent `status`) ------------------
    def context(self, mode: str | None = None, rng: random.Random | None = None, rejection_note: str = "") -> Context:
        rng = rng or random.Random()
        entries = self.entries()
        archive = self.archive(entries)
        if archive.global_best is None:
            raise RuntimeError("no scored candidate in the ledger; re-run `init`")
        mode = mode or self.choose_mode(entries, archive, rng)
        parents = [archive.select_parent(rng, self.config.exploit)]
        if mode == "merge":
            comp = archive.complementary()
            parents = [archive.global_best, rng.choice(comp)[0]]
        elif mode == "fix_losers":
            parents = [archive.global_best]
        diag = "\n\n".join(
            format_diagnostics(p.eval_result(self.problem.objective_split), archive, label=f"Parent #{p.id}: ")
            for p in parents)
        extra = {}
        since = self.steps_since_improvement(entries)
        if since >= self.config.patience:
            extra["plateau"] = since
        if mode == "merge":
            comp = archive.complementary()
            extra["complementary"] = {e.id: won for e, won in comp}
        falsified = [e for e in entries if e.verdict in (VERDICT_FALSIFIED, VERDICT_UNCONFIRMED)]
        if falsified:
            extra["falsified"] = [(e.id, e.verdict, e.hypothesis) for e in falsified[-8:]]
        return Context(mode=mode, parents=parents, parent_sources=[self.store.read_candidate(p) for p in parents],
                       diagnostics=diag, digest=format_digest(entries), rejection_note=rejection_note, extra=extra)

    # ----- submission: novelty gate -> cascade -> archive update -> ledger ---------------------
    def submit(self, source: str, hypothesis: str, mode: str, parent_ids: list[int], proposer: str = "agent",
               prompt_tokens: int = 0, completion_tokens: int = 0) -> Entry:
        t0 = time.perf_counter()
        entries = self.entries()
        archive = self.archive(entries)
        cid = self.store.next_id()
        entry = Entry(id=cid, parent_ids=parent_ids, mode=mode, hypothesis=hypothesis, status="pending",
                      proposer=proposer, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        entry.source_path = self.store.write_candidate(cid, source)

        violations = check_imports(source, self.problem.allowed_imports)
        prior = [(e.id, self.store.read_candidate(e)) for e in entries if e.source_path]
        verdict = check_novelty(source, prior, self.config.novelty_threshold)
        entry.novelty = verdict.to_dict()
        if violations:
            entry.status = STATUS_REJECTED_GUARD
            entry.verdict = VERDICT_UNTESTED
            entry.note = "disallowed in candidate: " + ", ".join(violations) + "; not evaluated"
        elif verdict.is_duplicate:
            entry.status = STATUS_REJECTED_DUPLICATE
            entry.verdict = VERDICT_UNTESTED
            entry.note = f"near-duplicate of #{verdict.nearest_id} (similarity {verdict.max_similarity:.3f}); not evaluated"
        else:
            self._cascade(entry, source)
            if entry.scored:
                improved_global, improved_instances = archive.improvement_of(entry)
                best = archive.global_best.objective
                entry.note = (f"objective {entry.objective:g} vs best {best:g} "
                              f"({100 * (best - entry.objective) / max(best, 1):+.1f}%)")
                if improved_global:
                    entry.confirmed = self._confirm(entry, source, incumbent=archive.global_best)
                    if entry.confirmed is False:
                        improved_global = False
                        inc = archive.global_best.eval_result(self.problem.confirm_split)
                        mine = entry.eval_result(self.problem.confirm_split)
                        entry.note += (f"; NOT confirmed on fresh '{self.problem.confirm_split}' instances "
                                       f"({mine.score:g} vs incumbent {inc.score:g}) — validate gain treated as noise/overfit")
                entry.improved_global = improved_global
                entry.improved_instances = improved_instances
                entry.status = STATUS_KEPT if (improved_global or improved_instances) else STATUS_EVALUATED
                if improved_global:
                    entry.verdict = VERDICT_SUPPORTED
                elif entry.confirmed is False:
                    entry.verdict = VERDICT_UNCONFIRMED
                elif improved_instances:
                    entry.verdict = VERDICT_PARTIAL
                else:
                    entry.verdict = VERDICT_FALSIFIED
            else:
                entry.verdict = VERDICT_INCONCLUSIVE
        entry.elapsed = time.perf_counter() - t0
        self.store.append(entry)
        return entry

    def _cascade(self, entry: Entry, source: str) -> None:
        """Evaluate split by split; stop early when a split fails or is worse than baseline."""
        for split in self.problem.splits:
            if split == "holdout" or split == self.problem.confirm_split:
                continue
            res = evaluate_in_subprocess(self.problem_name, self.problem, source, split)
            entry.evals[split] = res.to_dict()
            if not res.ok:
                entry.status = STATUS_FAILED if split == self.problem.objective_split else STATUS_REJECTED_SCREEN
                entry.objective = float("inf")
                entry.note = res.error or "; ".join(f"{i.name}: {i.error}" for i in res.instances if not i.valid)
                return
            if split != self.problem.objective_split and res.score > res.baseline:
                entry.status = STATUS_REJECTED_SCREEN
                entry.objective = float("inf")
                entry.note = f"worse than baseline on '{split}' ({res.score:g} > {res.baseline:g}); objective split skipped"
                return
            if split == self.problem.objective_split:
                entry.objective = res.score
                return

    def _confirm(self, entry: Entry, source: str, incumbent: Entry | None = None) -> bool | None:
        """Re-test a claimed new global best on fresh instances: confirmed unless it scores worse there."""
        split = self.problem.confirm_split
        if not split:
            return None
        res = evaluate_in_subprocess(self.problem_name, self.problem, source, split)
        entry.evals[split] = res.to_dict()
        if not res.ok:
            return False
        ref = incumbent.eval_result(split) if incumbent else None
        # a tie on fresh instances is not evidence of overfitting; only a regression refutes the gain
        return ref is None or not ref.ok or res.score <= ref.score

    def evaluate_holdout(self, entry: Entry) -> EvalResult:
        return evaluate_in_subprocess(self.problem_name, self.problem, self.store.read_candidate(entry), "holdout")

    @staticmethod
    def describe_entry(e: Entry) -> str:
        obj = "-" if e.objective is None else ("fail" if e.objective == float("inf") else f"{e.objective:g}")
        tag = " ** NEW GLOBAL BEST **" if e.improved_global else (f" (new best on {len(e.improved_instances)} instance(s))" if e.improved_instances else "")
        verdict = f" verdict={e.verdict}" if e.verdict else ""
        return f"#{e.id} [{e.mode}] {e.status}{verdict} objective={obj}{tag} tokens={e.tokens} — {e.hypothesis}\n    {e.note}"
