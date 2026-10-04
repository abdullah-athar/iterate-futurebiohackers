"""The research loop: propose -> novelty gate -> cascade evaluate -> archive -> ledger."""

from __future__ import annotations

import json
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
from .novelty import NoveltyVerdict, check_novelty
from .problem import EvalResult, Problem, get_problem
from .prompts import (
    MODES,
    Context,
    format_diagnostics,
    format_digest,
)
from .sandbox import evaluator_hash, get_evaluate


def evaluate_candidate(problem_name: str, source: str, budget_ms: int | None, evaluate=None) -> dict[str, dict]:
    """Cascade: cheap splits first, stop when one fails or is worse than baseline; once the objective
    split passes, also run the fresh-instance confirm split. Pure (no ledger access), so it runs
    the same in agent mode and on remote workers. `evaluate(problem_name, problem, source, split,
    budget_ms) -> EvalResult` defaults to `sandbox.get_evaluate()`."""
    evaluate = evaluate or get_evaluate()
    problem = get_problem(problem_name)
    evals: dict[str, dict] = {}
    for split in problem.splits:
        if split == "holdout" or split == problem.confirm_split:
            continue
        res = evaluate(problem_name, problem, source, split, budget_ms)
        evals[split] = res.to_dict()
        if not res.ok or (split != problem.objective_split and res.score > res.baseline):
            return evals
        if split == problem.objective_split:
            break
    if problem.confirm_split:
        evals[problem.confirm_split] = evaluate(problem_name, problem, source, problem.confirm_split, budget_ms).to_dict()
    return evals


@dataclass
class LoopConfig:
    problem: str = "median_string"
    novelty_threshold: float = 0.95
    max_novelty_attempts: int = 3
    exploit: float = 0.7
    patience: int = 4          # proposals without global improvement before 'plateau' (single agent)
    patience_generations: int = 1  # swarm: whole generations without a new best before 'plateau'
    ucb_c: float = 0.8
    time_budget_ms: int = 1000  # CPU budget per instance for one solve() call
    # ablation controls: restrict the bandit to these prompt modes (None = all). `--modes tune --exploit 1.0`
    # is the plain incumbent-only loop (no per-instance archive, merge or mode selection) to compare against.
    modes: list[str] | None = None
    # exploration-exploitation layer (swarm only, see autoresearch/exploration_exploitation.py)
    descriptors: bool = False
    desc_threshold: float = 0.3    # new_family: min descriptor distance to the elite archive to be evaluated
    drift_threshold: float = 0.3   # tune: descriptor drift from the parent that re-routes a child to new_family
    elite: int = 10                # elite archive = top N scored candidates by objective
    landscape: int = 25            # archive members shown to agents as descriptor + summary
    gap_fraction: float = 1 / 3    # share of new_family agents with a gap prompt
    describe_model: str = "sonnet"

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
    def create(cls, store: RunStore, config: LoopConfig, seed_source: str | None = None, evaluate=None) -> ResearchRun:
        if store.exists:
            raise FileExistsError(f"run already exists at {store.root}")
        store.create({**config.to_dict(), "evaluator_hash": evaluator_hash()})
        run = cls(store)
        source = seed_source or run.problem.seed_source()
        entry = Entry(id=0, parent_ids=[], mode="seed", hypothesis="Seed solver (starting point)", status=STATUS_SEED,
                      proposer="seed")
        entry.source_path = store.write_candidate(0, source)
        entry.evals = evaluate_candidate(run.problem_name, source, config.time_budget_ms, evaluate)
        run._apply_cascade(entry)
        if entry.scored:
            entry.improved_global = True
            entry.confirmed = run._confirmed(entry, None)
        store.append(entry)
        run.write_baselines(evaluate)
        return run

    def write_baselines(self, evaluate=None) -> dict:
        """Score the problem's classical (non-agent) solvers on the objective and holdout splits
        under the run's budget -> baselines.json (reference rows for the report and dashboard)."""
        names = getattr(self.problem, "baseline_solvers", ())
        evaluate = evaluate or get_evaluate()
        out = {}
        for name in names:
            src = self.problem.baseline_source(name)
            out[name] = {split: evaluate(self.problem_name, self.problem, src, split, self.config.time_budget_ms).to_dict()
                         for split in (self.problem.objective_split, "holdout")}
        if out:
            (self.store.root / "baselines.json").write_text(json.dumps(out, indent=1))
        return out

    def baselines(self) -> dict[str, dict[str, EvalResult]]:
        path = self.store.root / "baselines.json"
        if not path.exists():
            return {}
        return {name: {split: EvalResult.from_dict(d) for split, d in rec.items()}
                for name, rec in json.loads(path.read_text()).items()}

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

    def generations_since_improvement(self, entries: list[Entry]) -> int:
        gens = sorted({e.generation for e in entries if e.generation})
        improved = {e.generation for e in entries if e.generation and e.improved_global}
        n = 0
        for g in reversed(gens):
            if g in improved:
                break
            n += 1
        return n

    def plateau(self, entries: list[Entry]) -> bool:
        """Swarm runs count whole generations (16 flat proposals in one generation is one data point);
        single-agent runs count proposals."""
        if any(e.generation for e in entries):
            return self.generations_since_improvement(entries) >= self.config.patience_generations
        return self.steps_since_improvement(entries) >= self.config.patience

    def mode_scores(self, entries: list[Entry], archive: Archive) -> dict[str, float]:
        """UCB1 score per eligible prompt mode (inf = untried). After a plateau `tune` is not eligible."""
        allowed = [m for m in MODES if not self.config.modes or m in self.config.modes] or list(MODES)
        eligible = [m for m in allowed if m != "merge" or archive.complementary()] or [allowed[0]]
        if self.plateau(entries):
            eligible = [m for m in eligible if m != "tune"] or eligible
        stats = {m: [0, 0.0] for m in MODES}
        for e in entries:
            # a proposal the descriptor gate dropped unevaluated still cost an agent: a zero-reward try
            # (code duplicates and guard rejections stay uncounted, as without the layer)
            skipped = e.status in (STATUS_SEED, STATUS_REJECTED_DUPLICATE, STATUS_REJECTED_GUARD)
            if e.mode in stats and (not skipped or e.usage.get("gate_dropped")):
                stats[e.mode][0] += 1
                stats[e.mode][1] += 1.0 if e.improved_global else (0.5 if e.improved_instances else 0.0)
        total = sum(n for n, _ in stats.values()) or 1
        return {m: math.inf if stats[m][0] == 0 else
                stats[m][1] / stats[m][0] + self.config.ucb_c * math.sqrt(math.log(total) / stats[m][0])
                for m in eligible}

    def choose_mode(self, entries: list[Entry], archive: Archive, rng: random.Random) -> str:
        scores = self.mode_scores(entries, archive)
        untried = [m for m, v in scores.items() if v == math.inf]
        if untried:
            return untried[0] if not self.plateau(entries) else rng.choice(untried)
        return max(scores, key=scores.get)

    def pick_parents(self, mode: str, archive: Archive, rng: random.Random) -> list[Entry]:
        if mode == "merge":
            return [archive.global_best, rng.choice(archive.complementary())[0]]
        if mode == "fix_losers":
            return [archive.global_best]
        return [archive.select_parent(rng, self.config.exploit)]

    # ----- context for the proposer (agent `status` and swarm workspaces) ---------------------
    def context(self, mode: str | None = None, rng: random.Random | None = None, rejection_note: str = "",
                parent_ids: list[int] | None = None) -> Context:
        rng = rng or random.Random()
        entries = self.entries()
        archive = self.archive(entries)
        if archive.global_best is None:
            raise RuntimeError("no scored candidate in the ledger; re-run `init`")
        mode = mode or self.choose_mode(entries, archive, rng)
        if parent_ids:
            by_id = {e.id: e for e in entries}
            parents = [by_id[i] for i in parent_ids]
        else:
            parents = self.pick_parents(mode, archive, rng)
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
            extra["falsified"] = [(e.id, e.verdict, e.hypothesis) for e in falsified[-20:]]
        return Context(mode=mode, parents=parents, parent_sources=[self.store.read_candidate(p) for p in parents],
                       diagnostics=diag, digest=format_digest(entries), rejection_note=rejection_note, extra=extra)

    # ----- submission: guard + novelty gate -> evaluation -> archive update -> ledger -----------------
    def submit(self, source: str, hypothesis: str, mode: str, parent_ids: list[int], proposer: str = "agent",
               prompt_tokens: int = 0, completion_tokens: int = 0, evaluate=None) -> Entry:
        """Agent mode: check, evaluate and record one candidate (single process)."""
        t0 = time.perf_counter()
        pre = self.precheck(source)
        evals = {} if pre[0] or pre[1].is_duplicate else evaluate_candidate(
            self.problem_name, source, self.config.time_budget_ms, evaluate)
        return self.record(source, hypothesis, mode, parent_ids, proposer, evals, pre,
                           elapsed=time.perf_counter() - t0, prompt_tokens=prompt_tokens,
                           completion_tokens=completion_tokens)

    def precheck(self, source: str, extra_prior: list[tuple[int, str]] = (),
                 threshold: float | None = None) -> tuple[list[str], NoveltyVerdict]:
        """Import guard + novelty gate against every recorded candidate (and `extra_prior`)."""
        violations = check_imports(source, self.problem.allowed_imports)
        prior = [(e.id, self.store.read_candidate(e)) for e in self.entries() if e.source_path]
        return violations, check_novelty(source, prior + list(extra_prior),
                                         self.config.novelty_threshold if threshold is None else threshold)

    def record(self, source: str, hypothesis: str, mode: str, parent_ids: list[int], proposer: str,
               evals: dict[str, dict], pre: tuple[list[str], NoveltyVerdict], **fields) -> Entry:
        """Turn precheck + evaluation results into a ledger entry. The only place proposals are written."""
        archive = self.archive()
        cid = self.store.next_id()
        entry = Entry(id=cid, parent_ids=parent_ids, mode=mode, hypothesis=hypothesis, status="pending",
                      proposer=proposer, **fields)
        entry.source_path = self.store.write_candidate(cid, source)
        violations, novelty = pre
        entry.novelty = novelty.to_dict()
        if violations:
            entry.status = STATUS_REJECTED_GUARD
            entry.verdict = VERDICT_UNTESTED
            entry.note = "disallowed in candidate: " + ", ".join(violations) + "; not evaluated"
        elif novelty.is_duplicate:
            entry.status = STATUS_REJECTED_DUPLICATE
            entry.verdict = VERDICT_UNTESTED
            entry.note = entry.note or f"near-duplicate of #{novelty.nearest_id} (similarity {novelty.max_similarity:.3f}); not evaluated"
        else:
            entry.evals = evals
            self._apply_cascade(entry)
            if entry.scored:
                improved_global, improved_instances = archive.improvement_of(entry)
                best = archive.global_best.objective
                entry.note = (f"objective {entry.objective:g} vs best {best:g} "
                              f"({100 * (best - entry.objective) / max(best, 1):+.1f}%)")
                if improved_global:
                    entry.confirmed = self._confirmed(entry, archive.global_best)
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
        self.store.append(entry)
        return entry

    def _apply_cascade(self, entry: Entry) -> None:
        """Set objective/status from the cascade evals (see `evaluate_candidate`)."""
        for split in self.problem.splits:
            if split == "holdout" or split == self.problem.confirm_split or split not in entry.evals:
                continue
            res = entry.eval_result(split)
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

    def _confirmed(self, entry: Entry, incumbent: Entry | None) -> bool | None:
        """A claimed new global best is confirmed unless it scores worse than the incumbent on fresh instances."""
        split = self.problem.confirm_split
        if not split:
            return None
        res = entry.eval_result(split)
        if res is None or not res.ok:
            return False
        ref = incumbent.eval_result(split) if incumbent else None
        # a tie on fresh instances is not evidence of overfitting; only a regression refutes the gain
        return ref is None or not ref.ok or res.score <= ref.score

    def evaluate_holdout(self, entry: Entry) -> EvalResult:
        return get_evaluate()(self.problem_name, self.problem, self.store.read_candidate(entry), "holdout",
                              self.config.time_budget_ms)

    @staticmethod
    def describe_entry(e: Entry) -> str:
        obj = "-" if e.objective is None else ("fail" if e.objective == float("inf") else f"{e.objective:g}")
        tag = " ** NEW GLOBAL BEST **" if e.improved_global else (f" (new best on {len(e.improved_instances)} instance(s))" if e.improved_instances else "")
        verdict = f" verdict={e.verdict}" if e.verdict else ""
        return f"#{e.id} [{e.mode}] {e.status}{verdict} objective={obj}{tag} tokens={e.tokens} — {e.hypothesis}\n    {e.note}"
