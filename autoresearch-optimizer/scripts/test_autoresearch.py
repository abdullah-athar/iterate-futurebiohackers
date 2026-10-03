"""Tests for the autoresearch loop: novelty gate, import guard, archive, agent-mode run."""

import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from autoresearch.archive import Archive
from autoresearch.guard import check_imports
from autoresearch.ledger import (
    STATUS_KEPT,
    STATUS_REJECTED_DUPLICATE,
    STATUS_REJECTED_GUARD,
    STATUS_REJECTED_SCREEN,
    RunStore,
)
from autoresearch.loop import LoopConfig, ResearchRun
from autoresearch.novelty import check_novelty, normalize
from autoresearch.problem import get_problem
from autoresearch.report import render

import autoresearch_fixtures as mp


def test_novelty_gate():
    print("Testing novelty gate...")
    seed = get_problem("median_string").seed_source()
    dup = check_novelty(mp.SEED_DUPLICATE, [(0, seed)])
    assert dup.is_duplicate, dup
    assert dup.nearest_id == 0
    fresh = check_novelty(mp.CONSENSUS_LOCAL_SEARCH, [(0, seed)])
    assert not fresh.is_duplicate, fresh
    # alpha-renaming of locals must not change the normalised form
    a = normalize("def solve(instance):\n    xs = instance.strings\n    return min(xs)\n")
    b = normalize("def solve(instance):\n    ys = instance.strings\n    return min(ys)\n")
    assert a == b
    print("  Novelty gate test passed!")


def test_import_guard():
    print("Testing import guard...")
    allowed = ("median_string.metrics",)
    assert check_imports("import random\nfrom median_string.metrics import sum_distance\n", allowed) == []
    bad = check_imports("import os, subprocess\nfrom median_string.benchmarks import x\nexec('1')\n", allowed)
    assert len(bad) == 4, bad
    assert check_imports("def solve(instance:\n", allowed) == []  # syntax errors are the evaluator's job
    print("  Import guard test passed!")


def test_agent_run():
    print("Testing agent-mode research run (evaluates real candidates, ~1 min)...")
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-test-"))
    try:
        store = RunStore(tmp / "run")
        run = ResearchRun.create(store, LoopConfig(problem="median_string"))
        seed_entry = run.entries()[0]
        assert seed_entry.confirmed is True and "confirm" in seed_entry.evals
        proposals = [
            (mp.SEED_DUPLICATE, "cosmetic rewrite of the seed", "tune"),
            (mp.CONSENSUS_LOCAL_SEARCH, "consensus start + full re-scoring local search", "tune"),
            (mp.FAST_DESCENT, "steepest descent with incremental DP over sub/ins/del", "tune"),
            (mp.BROKEN, "syntax error", "fix_losers"),
        ]
        entries = [run.submit(src, hyp, mode, [run.archive().global_best.id], proposer="test")
                   for src, hyp, mode in proposals]
        statuses = [e.status for e in entries]
        # duplicate caught; slow solver over the 1000 ms budget; fast solver kept; syntax error caught
        assert statuses == [STATUS_REJECTED_DUPLICATE, "failed", STATUS_KEPT, STATUS_REJECTED_SCREEN], statuses
        assert "over budget" in entries[1].note, entries[1].note
        assert all(e.verdict for e in entries), [e.verdict for e in entries]
        assert entries[2].confirmed is True and entries[2].verdict == "supported"
        archive = Archive.build(run.entries(), run.problem.objective_split)
        assert archive.global_best is not None and archive.global_best.objective < seed_entry.objective
        assert len(archive.front) >= 1
        report = render(run, holdout=False)
        assert "Pareto front" in report and "tokens" in report
        guarded = run.submit("import os\ndef solve(instance):\n    return instance.strings[0]\n", "cheat", "tune", [0])
        assert guarded.status == STATUS_REJECTED_GUARD and "confirm" not in guarded.evals
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Agent-mode run test passed!")


def test_swarm():
    print("Testing swarm generations with a fake proposer (local evaluation)...")
    from autoresearch import swarm

    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-swarm-"))
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string"))
        sources = [mp.FAST_DESCENT, mp.SEED_DUPLICATE, ""]

        def propose_many(run, gen, assignments):
            return [{"assignment": a, "source": sources[a.worker % 3], "hypothesis": f"w{a.worker}",
                     "usage": {"model": "fake", "seconds": 1.0, "outcome": "ok", "cost_usd": 0.01}} for a in assignments]

        lines = []
        emit = swarm.Events(run, 0.0, log=lines.append)
        swarm.run_swarm(run, 3, 1, 1e9, propose_many, swarm.local_evaluator("median_string", 1000), emit,
                        max_generations=2)
        entries = run.entries()
        assert [e.id for e in entries] == list(range(len(entries))), [e.id for e in entries]
        assert [e.status for e in entries if e.generation == 1] == [STATUS_KEPT, STATUS_REJECTED_DUPLICATE]
        assert all(e.status == STATUS_REJECTED_DUPLICATE for e in entries if e.generation == 2)
        kinds = [json.loads(l)["type"] for l in (run.store.root / "events.jsonl").read_text().splitlines()]
        assert kinds.count("gen_end") == 2 and kinds[-1] == "run_end", kinds
        assert (run.store.root / "holdout.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Swarm test passed!")


if __name__ == "__main__":
    random.seed(0)
    test_novelty_gate()
    test_import_guard()
    test_agent_run()
    test_swarm()
    print("\nAll autoresearch tests passed!")
