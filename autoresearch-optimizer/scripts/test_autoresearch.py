"""Tests for the autoresearch loop: novelty gate, import guard, candidate/scorer boundary, archive, agent-mode run,
hypothesis gate, dashboard lineage."""

import json
import os
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
from autoresearch.hypothesis_gate import _parse_decisions, lexical_judge
from autoresearch.ledger import (
    STATUS_KEPT,
    STATUS_REJECTED_DUPLICATE,
    STATUS_REJECTED_GUARD,
    STATUS_REJECTED_HYPOTHESIS,
    STATUS_REJECTED_SCREEN,
    RunStore,
)
from autoresearch.loop import LoopConfig, ResearchRun
from autoresearch.novelty import check_novelty, normalize
from autoresearch.problem import get_problem
from autoresearch.report import render
from autoresearch.sandbox import evaluate_in_subprocess

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


HONEST = "def solve(instance):\n    return instance.strings[0]\n"
PEEK_ANSWER = "def solve(instance):\n    return instance.planted_consensus\n"
NO_SECRETS = ("def solve(instance):\n"
              "    assert instance.planted_consensus is None and instance.known_best_score is None\n"
              "    assert not instance.metadata and instance.description == ''\n"
              "    return instance.strings[0]\n")
MUTATE_INPUTS = "def solve(instance):\n    instance.strings[:] = instance.strings[:1]\n    return instance.strings[0]\n"
TAMPER_SCORER = ("import median_string.metrics as m\n\ndef solve(instance):\n"
                 "    m.levenshtein_distance = lambda a, b: 0\n    m.hamming_distance = lambda a, b: 0\n"
                 "    return instance.strings[0]\n")
NOT_A_STRING = "def solve(instance):\n    return 12345\n"
READ_SECRET_ENV = ("import os\n\ndef solve(instance):\n"
                   "    return instance.strings[0] if 'AUTORESEARCH_CONFIRM_SEED' not in os.environ else ''\n")


def test_evaluator_boundary():
    """Candidate code runs in its own process on a copy of the inputs; the parent scores the originals.
    Each probe below produced a false score on the previous single-process evaluator."""
    print("Testing the candidate/scorer boundary...")
    problem = get_problem("median_string")

    def ev(src):
        return evaluate_in_subprocess("median_string", problem, src, "screen", 1000)

    honest = ev(HONEST)
    assert honest.ok and honest.score > honest.baseline > 0, honest
    # 1. the planted answer, its score and the generator metadata are not in what the candidate sees
    peek = ev(PEEK_ANSWER)
    assert not peek.ok and all("must be a string" in i.error for i in peek.instances), peek
    assert ev(NO_SECRETS).score == honest.score
    # 2. mutating the inputs changes nothing: score and baseline come from the evaluator's own copy
    mut = ev(MUTATE_INPUTS)
    assert mut.ok and (mut.score, mut.baseline) == (honest.score, honest.baseline), mut
    # 3. monkeypatching the distance functions inside the candidate's process cannot reach the scorer
    tam = ev(TAMPER_SCORER)
    assert tam.ok and (tam.score, tam.baseline) == (honest.score, honest.baseline), tam
    # 4. a non-string answer is invalid (penalty), not a crash of the harness
    bad = ev(NOT_A_STRING)
    assert not bad.ok and bad.score == bad.baseline + 1000 * len(bad.instances), bad
    # 5. the hidden-suite seeds are stripped from the candidate's environment
    os.environ["AUTORESEARCH_CONFIRM_SEED"] = "4242"
    try:
        env = ev(READ_SECRET_ENV)
    finally:
        del os.environ["AUTORESEARCH_CONFIRM_SEED"]
    assert env.ok and env.score == honest.score, env
    # 6. budget enforcement and per-instance CPU accounting still work across the process boundary
    slow = ev(mp.SLOW_ON_VALIDATE.replace("> 18", ">= 0"))
    assert not slow.ok and all("over budget" in i.error and i.cpu_ms > 1000 for i in slow.instances), slow
    print("  Boundary test passed!")


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
            (mp.SLOW_ON_VALIDATE, "set median, but slow on the objective split", "tune"),
            (mp.FAST_INDEL_SEARCH, "set median + substitution/insertion/deletion search", "tune"),
            (mp.BROKEN, "syntax error", "fix_losers"),
        ]
        entries = [run.submit(src, hyp, mode, [run.archive().global_best.id], proposer="test")
                   for src, hyp, mode in proposals]
        statuses = [e.status for e in entries]
        # duplicate caught; solver over the 1000 ms budget on validate fails; fast solver kept; syntax error caught
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


def test_hypothesis_gate():
    print("Testing hypothesis gate (lexical judge, judge-answer parsing)...")
    judge = lexical_judge()
    prior = [("#3", "simulated annealing over insert and delete moves from the set median")]
    new = [("w00", "Simulated annealing with insert/delete moves starting from the set median"),
           ("w01", "bit-parallel Myers distance kernel to evaluate more neighbours within the CPU budget"),
           ("w02", "Myers bit-parallel distance kernel so more neighbours are evaluated within CPU budget")]
    verdicts, usage = judge(new, prior)
    assert [v.keep for v in verdicts] == [False, True, False], verdicts
    assert verdicts[0].similar_to == "#3" and verdicts[2].similar_to == "w01" and usage == {}

    labels = ["w00", "w01", "w02", "w03"]
    answer = ('noise {"decisions": [{"id": "w00", "duplicate_of": "#7", "reason": "same SA"}, '
              '{"id": "w01", "duplicate_of": null}, {"id": "w02", "duplicate_of": "w01", "reason": "same kernel"}, '
              '{"id": "w03", "duplicate_of": "w03"}]} trailing')
    verdicts = _parse_decisions(answer, labels)
    # a ledger id or an earlier kept proposal rejects; a self/forward reference keeps the idea
    assert [v.keep for v in verdicts] == [False, True, False, True], verdicts
    assert verdicts[2].similar_to == "w01" and verdicts[2].reason == "same kernel"

    from autoresearch.swarm import _first_line, _merge_usage
    hyp = {"model": "m", "seconds": 2.0, "cost_usd": 0.02, "prompt_tokens": 5000, "completion_tokens": 50, "gate_cost_usd": 0.005}
    code = {"model": "m", "seconds": 40.0, "cost_usd": 0.10, "prompt_tokens": 90000, "completion_tokens": 3000,
            "num_turns": 8, "outcome": "ok"}
    u = _merge_usage(hyp, code)
    assert abs(u["cost_usd"] - 0.125) < 1e-9 and u["hyp_cost_usd"] == 0.02 and u["gate_cost_usd"] == 0.005
    assert u["prompt_tokens"] == 95000 and u["outcome"] == "ok"
    assert abs(_merge_usage(hyp)["cost_usd"] - 0.025) < 1e-9
    assert _first_line("\n- **Use Myers bit-parallel distance**\nbecause...") == "Use Myers bit-parallel distance"
    print("  Hypothesis gate test passed!")


def test_swarm():
    print("Testing swarm generations with a fake proposer (local evaluation)...")
    from autoresearch import swarm

    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-swarm-"))
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string"))
        sources = [mp.FAST_INDEL_SEARCH, mp.SEED_DUPLICATE, ""]

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


def test_swarm_hypothesis_first():
    print("Testing swarm recording of hypothesis-gate rejections...")
    from autoresearch import swarm

    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-swarm-hyp-"))
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string"))

        def propose_many(run, gen, assignments):
            u = {"model": "fake", "seconds": 1.0, "outcome": "ok", "cost_usd": 0.03, "hyp_cost_usd": 0.01}
            out = [{"assignment": assignments[0], "source": mp.FAST_INDEL_SEARCH, "hypothesis": "fast indel search", "usage": u}]
            for a in assignments[1:]:
                out.append({"assignment": a, "source": "", "hypothesis": "fast indel search again",
                            "usage": {**u, "outcome": "rejected_hypothesis", "cost_usd": 0.01},
                            "gate": {"keep": False, "similar_to": "w00", "reason": "same idea"}})
            return out

        lines = []
        emit = swarm.Events(run, 0.0, log=lines.append)
        swarm.run_swarm(run, 3, 1, 1e9, propose_many, swarm.local_evaluator("median_string", 1000), emit,
                        max_generations=1)
        gen1 = [e for e in run.entries() if e.generation == 1]
        assert [e.status for e in gen1] == [STATUS_KEPT, STATUS_REJECTED_HYPOTHESIS, STATUS_REJECTED_HYPOTHESIS]
        assert all(not e.source_path and "same idea as w00" in e.note for e in gen1[1:])
        # repeated ideas are not evidence for the mode bandit, and the report counts them apart
        assert run.steps_since_improvement(run.entries()) == 0
        text = render(run)
        assert "1 evaluated" in text and "hypothesis gate: 2 of 3 ideas stopped" in text, text
        assert any("1 hypothesis-gate" not in l and "2 hypothesis-gate" in l for l in lines), lines
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Swarm hypothesis-first test passed!")


def test_swarm_model_schedule():
    print("Testing the per-generation model schedule (--model sonnet,opus)...")
    from autoresearch import swarm

    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-swarm-sched-"))
    real, seen = swarm.run_claude, []
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string"))
        swarm.run_claude = lambda ws, turn_s, model, *a, **k: seen.append(model) or {"model": model, "seconds": 0.0, "outcome": "ok"}
        propose = swarm.claude_proposer("sonnet,opus", 1, "local", None, lambda *a, **k: None)
        for gen in (1, 2, 3):
            propose(run, gen, [swarm.Assignment(0, "tune", [0])])
        assert seen == ["sonnet", "opus", "opus"], seen  # the last model is kept after the schedule ends
    finally:
        swarm.run_claude = real
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Model schedule test passed!")


def test_viz_lineage():
    print("Testing the dashboard's idea-lineage chart (merges, duplicates, disjoint trees)...")
    from autoresearch_viz.html import render
    from autoresearch_viz.load import load_run

    rows = [
        {"id": 0, "parent_ids": [], "mode": "seed", "status": "seed", "objective": 100, "improved_global": True},
        {"id": 1, "parent_ids": [0], "mode": "tune", "status": "kept", "objective": 90, "improved_global": True, "verdict": "supported", "generation": 1},
        {"id": 2, "parent_ids": [0], "mode": "new_family", "status": "evaluated", "objective": 95, "verdict": "falsified", "generation": 1},
        {"id": 3, "parent_ids": [0], "mode": "tune", "status": "rejected_duplicate", "verdict": "untested", "generation": 1,
         "novelty": {"nearest_id": 1, "max_similarity": 0.99}},
        {"id": 4, "parent_ids": [1, 2], "mode": "merge", "status": "kept", "objective": 80, "improved_global": True, "generation": 2,
         "hypothesis": "combine <both> & \"quote\""},
        {"id": 5, "parent_ids": [4], "mode": "fix_losers", "status": "rejected_duplicate", "generation": 3},
        {"id": 6, "parent_ids": [4, 99], "mode": "tune", "status": "failed", "generation": 3},
        {"id": 7, "parent_ids": [0], "mode": "new_family", "status": "evaluated", "objective": 97},
        {"id": 8, "parent_ids": [7], "mode": "tune", "status": "evaluated", "objective": 96},
    ]
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-viz-"))
    try:
        (tmp / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        html = render([load_run(tmp)])
        card = html[html.index("<h2>Idea lineage</h2>"):html.index("<h2>Research trajectory</h2>")]
        ideas = card[card.index('data-group="linview0" data-key="ideas"'):card.index('data-group="linview0" data-key="evaluated"')]
        assert ideas.count('class="ln') == 8 and 'data-id="0"' not in ideas, "seed must be hidden by default"
        assert ideas.count('class="le') == 5, ideas.count('class="le')  # 1→4, 2→4, 4→5, 4→6, 7→8
        assert "one-off idea" in ideas and "ideas from #1, #2" in ideas and "ideas from #7" in ideas
        assert "★ #4 best" in ideas and "&lt;both&gt;" in ideas
        evaluated = card[card.index('data-group="linview0" data-key="evaluated"'):card.index('data-group="linview0" data-key="seed"')]
        assert evaluated.count('class="ln') == 6 and "duplicates hidden" in evaluated
        with_seed = card[card.index('data-group="linview0" data-key="seed"'):]
        assert with_seed.count('class="ln') == 9 and "9 ideas from #0" in with_seed
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Lineage chart test passed!")


def test_viz_quality_efficiency():
    print("Testing the dashboard's quality vs efficiency card and cost axis...")
    from autoresearch_viz.html import render
    from autoresearch_viz.load import load_run

    def ev(score):
        return {"split": "validate", "score": score, "baseline": 100, "instances": []}

    def run(dir_, best, cost):
        rows = [{"id": 0, "parent_ids": [], "mode": "seed", "status": "seed", "objective": 90, "evals": {"validate": ev(90)},
                 "timestamp": 1.0},
                {"id": 1, "parent_ids": [0], "mode": "tune", "status": "kept", "objective": best, "improved_global": True,
                 "evals": {"validate": ev(best)}, "timestamp": 61.0, "usage": {"cost_usd": cost}, "proposer": "claude-code:x"}]
        dir_.mkdir()
        (dir_ / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return load_run(dir_)

    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-viz-qe-"))
    try:
        strong, cheap = run(tmp / "strong", 70, 2.0), run(tmp / "cheap", 80, 0.5)
        strong.label, cheap.label = "Strong", "Cheap"
        html = render([strong, cheap])
        card = html[html.index("<h2>Quality vs efficiency</h2>"):html.index("<h2>Research progress</h2>")]
        assert "Quality<span class='qe-win'> · Strong wins</span>" in card and "Efficiency<span class='qe-win'> · Cheap wins</span>" in card
        assert "+22.2% ★" in card and "$0.50 ★" in card and "20 ★" in card  # gain, cost, points per dollar (10 / 0.5)
        assert 'data-key="cost">vs cost</button>' in html
        assert "Cheap end · $0.50" in html and "Strong end · $2.00" in html, "each run's real end is marked on the cost axis"
        # one show/hide button per run, and every bar row is tagged so the page can hide it and re-rank the rest
        assert "<button class='on' aria-pressed='true' data-run='Strong'>" in html and "data-run='Cheap'>" in html
        assert card.count('<g class="mb" data-run="Cheap"') >= 4 and "class='card qe-card' data-runs=" in html
        assert "class='runtoggles'" not in render([strong])
        # a model schedule: the hand-over is marked on the curve with the time and cost spent so far
        rows = [json.loads(l) for l in (tmp / "strong" / "ledger.jsonl").read_text().splitlines()]
        rows.append({**rows[1], "id": 2, "parent_ids": [1], "objective": 65, "evals": {"validate": ev(65)}, "timestamp": 121.0,
                     "proposer": "claude-code:opus"})
        rows[1]["proposer"] = "claude-code:sonnet"
        (tmp / "sched").mkdir()
        (tmp / "sched" / "ledger.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        sched = render([load_run(tmp / "sched")])
        assert ">Sonnet → Opus</text>" in sched and "1.0m into the run, $2.00 spent" in sched
        assert "A ring marks" not in html, "no ring note without a model switch"
        assert "<h2>Quality vs efficiency</h2>" not in render([strong]), "the card needs two runs"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Quality vs efficiency card test passed!")


if __name__ == "__main__":
    random.seed(0)
    test_novelty_gate()
    test_import_guard()
    test_evaluator_boundary()
    test_agent_run()
    test_hypothesis_gate()
    test_swarm()
    test_swarm_hypothesis_first()
    test_swarm_model_schedule()
    test_viz_lineage()
    test_viz_quality_efficiency()
    print("\nAll autoresearch tests passed!")
