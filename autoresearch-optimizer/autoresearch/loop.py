"""The autoresearch loop. This is the only place that knows how the components fit together."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .core import Budget, Candidate, CandidateEvaluator, Memory, Proposer, Record, ResearchContext, Result, Selector, SuiteSpec


class IterationBudget:
    """Stop after n proposals (the seed evaluation does not count)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def exhausted(self, memory: Memory) -> bool:
        return len(memory.records) - 1 >= self.n


@dataclass
class RunReport:
    champion: Record
    heldout: Result

    def to_dict(self) -> dict:
        return {
            "champion_id": self.champion.candidate.id,
            "champion_iteration": self.champion.iteration,
            "search": self.champion.result.to_dict(),
            "heldout": self.heldout.to_dict(),
        }


def run_loop(
    seed: Candidate,
    proposer: Proposer,
    evaluator: CandidateEvaluator,
    selector: Selector,
    memory: Memory,
    budget: Budget,
    search: SuiteSpec,
    heldout: SuiteSpec,
    history_k: int = 10,
) -> RunReport:
    seed_result = evaluator.evaluate(seed, search)
    if not seed_result.valid:
        raise RuntimeError(f"Seed candidate failed on {search}: {seed_result.error}")
    memory.record(Record(0, seed, seed_result, accepted=True))
    _log(memory.records[-1])

    iteration = 0
    while not budget.exhausted(memory):
        iteration += 1
        ctx = ResearchContext(iteration, selector.parents(memory), memory.summary(history_k), search)
        try:
            candidate = proposer.propose(ctx)
            result = evaluator.evaluate(candidate, search)
        except Exception as e:  # keep long runs alive through API or proposer errors
            candidate = Candidate(code="", rationale="proposer error")
            result = Result(candidate.id, str(search), float("inf"), valid=False, error=f"{type(e).__name__}: {e}")
        memory.record(Record(iteration, candidate, result, accepted=selector.accept(memory, result)))
        _log(memory.records[-1])

    champion = memory.champion()
    report = RunReport(champion, evaluator.evaluate(champion.candidate, heldout))
    print(f"\nChampion {champion.candidate.id} (iter {champion.iteration}): "
          f"search {champion.result.score} vs baseline {champion.result.baseline_score} | "
          f"held-out {report.heldout.score} vs baseline {report.heldout.baseline_score}")
    return report


def _log(rec: Record) -> None:
    status = "ACCEPT" if rec.accepted else ("reject" if rec.result.valid else "FAIL  ")
    print(f"[iter {rec.iteration:3d}] {status} score={rec.result.score} ({rec.result.elapsed:.1f}s) "
          f"{rec.candidate.rationale[:80]!r}", flush=True)
    if rec.result.error and not rec.result.valid:
        print(f"           error: {rec.result.error.splitlines()[-1][:160]}", flush=True)


def save_report(report: RunReport, path) -> None:
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
