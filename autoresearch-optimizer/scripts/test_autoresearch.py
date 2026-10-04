"""Tests for the autoresearch loop: novelty gate, import guard, candidate/scorer boundary, archive, agent-mode run."""

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
        # the empty session (worker 2) is kept as a no_output try instead of vanishing with its cost
        assert [e.status for e in entries if e.generation == 1] == [STATUS_KEPT, STATUS_REJECTED_DUPLICATE, "no_output"]
        assert [e.status for e in entries if e.generation == 2] == [STATUS_REJECTED_DUPLICATE] * 2 + ["no_output"]
        assert entries[2].novelty.get("nearest_id") in (0, 1), entries[2].novelty  # ids line up with empty slots
        kinds = [json.loads(l)["type"] for l in (run.store.root / "events.jsonl").read_text().splitlines()]
        assert kinds.count("gen_end") == 2 and kinds[-1] == "run_end", kinds
        assert (run.store.root / "holdout.json").exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Swarm test passed!")


def _ee_swarm(tmp: Path, fail: tuple[str, ...] = ()):
    """One 6-agent generation with the layer on (fake proposer and describer); `fail` = fixture files whose
    describe call raises. Returns (run, assignments seen by the proposer, the descriptor_gate event, gen-1 entries)."""
    from autoresearch import exploration_exploitation as ee
    from autoresearch import swarm
    from autoresearch.descriptors import Descriptor

    fixtures = repo_root / "scripts" / "descriptor_fixtures"
    failing = {(fixtures / f).read_text() for f in fail}
    seed = get_problem("median_string").seed_source()
    hill = ("hill climbing", ["substitution moves", "set median initialisation", "first-improvement selection"])
    by_worker = [  # bandit gives 2 agents each to tune, fix_losers, new_family -> workers in that order
        (seed.replace("range(50)", "range(60)"), hill),                                            # tune: keeps descriptor
        ((fixtures / "editop_voting.py").read_text(), ("consensus voting", ["edit-operation voting"])),  # tune: drifts
        ((fixtures / "positional_vote.py").read_text(), None),                                     # fix_losers
        (mp.FAST_INDEL_SEARCH, None),                                                              # fix_losers
        ((fixtures / "hill_climb.py").read_text(), hill),                                          # new_family: same as seed
        ((fixtures / "center_star_consensus.py").read_text(), ("center-star alignment", ["column-wise majority vote"])),
    ]
    labels = {seed: hill} | {src: lab for src, lab in by_worker if lab}
    seen_assignments = []

    def fake_describe(source, vocab, problem, **kw):
        if source in failing:
            raise RuntimeError("describe failed: fake")
        core, mids = labels.get(source, ("greedy construction", ["column-wise majority vote"]))
        return Descriptor([(core, 6)] + [(t, 3) for t in mids], summary=core)

    def propose_many(run, gen, assignments):
        seen_assignments.extend(assignments)
        return [{"assignment": a, "source": by_worker[a.worker][0], "hypothesis": f"w{a.worker}",
                 "usage": {"model": "fake", "seconds": 1.0, "outcome": "ok", "cost_usd": 0.01}} for a in assignments]

    real_describe, ee.describe = ee.describe, fake_describe
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string", descriptors=True))
        emit = swarm.Events(run, 0.0, log=lambda line: None)
        swarm.run_swarm(run, 6, 1, 1e9, propose_many, swarm.local_evaluator("median_string", 1000), emit,
                        max_generations=1)
    finally:
        ee.describe = real_describe
    events = [json.loads(l) for l in (run.store.root / "events.jsonl").read_text().splitlines()]
    gate = next(e for e in events if e["type"] == "descriptor_gate")
    return run, seen_assignments, gate, [e for e in run.entries() if e.generation == 1]


def test_exploration_exploitation():
    print("Testing the exploration-exploitation layer in the swarm (fake proposer and describer, local evaluation)...")
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-ee-"))
    try:
        run, seen_assignments, g, gen1 = _ee_swarm(tmp)
        modes = [a.mode for a in seen_assignments]
        assert modes == ["tune", "tune", "fix_losers", "fix_losers", "new_family", "new_family"], modes
        assert all("Descriptor contract" in a.notes for a in seen_assignments if a.mode == "tune")
        assert all("Research landscape" in a.notes for a in seen_assignments)
        branch = [e.usage.get("branch") for e in gen1]
        assert branch == ["K", "K->L", None, None, "L", "L"], branch
        assert gen1[0].evals, "tune child that kept its descriptor must be evaluated"
        assert gen1[1].evals and gen1[5].evals, [e.note for e in gen1]
        dup = gen1[4]  # hill-climb descriptor identical to the seed's
        assert dup.status == STATUS_REJECTED_DUPLICATE and "not selected by max-min" in dup.note, dup.note
        assert dup.usage["min_dist"] < run.config.desc_threshold
        assert all(e.usage["min_dist"] >= run.config.desc_threshold for e in (gen1[1], gen1[5]))
        assert (g["tune_kept"], g["drifted"], g["pool"], g["slots"], g["selected"], g["undescribed"]) == (1, 1, 3, 3, 2, 0), g
        events = [json.loads(l) for l in (run.store.root / "events.jsonl").read_text().splitlines()]
        assert next(e for e in events if e["type"] == "gen_end")["vocab"] > 0
        # the bandit counts the gate-dropped proposal as a zero-reward new_family try
        assert dup.usage.get("gate_dropped") is True
        entries = run.entries()
        score = lambda: run.mode_scores(entries, run.archive(entries))["new_family"]  # noqa: E731
        counted = score()
        next(e for e in entries if e.id == dup.id).usage.pop("gate_dropped")
        assert score() == counted  # default: every session is a try, flag or not
        run.config.bandit_counts_all_attempts = False  # legacy counting: only the gate flag keeps it
        assert counted < score()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Exploration-exploitation test passed!")


def test_gate_describe_failure():
    print("Testing that a candidate whose describe call fails is evaluated, not dropped...")
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-ee-"))
    try:
        run, _, g, gen1 = _ee_swarm(tmp, fail=("center_star_consensus.py",))
        star = gen1[5]  # new_family, describe raised
        assert star.status != STATUS_REJECTED_DUPLICATE and star.evals, (star.status, star.note)
        assert star.usage["descriptor"] is None and not star.usage.get("gate_dropped")
        assert (g["pool"], g["selected"], g["undescribed"]) == (3, 1, 1), g
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Describe-failure test passed!")

def test_lineage_parents():
    import autoresearch.exploration_exploitation as ee
    from autoresearch.ledger import Entry

    def mk(i, parent, branch, obj):
        return Entry(id=i, parent_ids=[parent] if parent is not None else [], mode="x", hypothesis="",
                     status="scored", objective=obj, usage={"branch": branch} if branch else {})
    # 0 seed; 1 founds lineage A (L); 2, 3 refine it (K); 4 drifted off 2 (K->L) founds lineage B
    es = [mk(0, None, None, 9), mk(1, 0, "L", 5), mk(2, 1, "K", 3), mk(3, 2, "K", 1), mk(4, 2, "K->L", 2)]
    by_id = {e.id: e for e in es}
    assert [ee.lineage_root(e, by_id) for e in es] == [0, 1, 1, 1, 4]
    elites = [(e, None) for e in sorted(es, key=lambda e: e.objective)]  # 3, 4, 2, 1, 0
    assert [p.id for p, _ in ee.lineage_parents(elites, es, 3)] == [3, 4, 0]
    print("  Lineage-parents test passed!")


def test_gap_directions():
    import autoresearch.exploration_exploitation as ee
    from autoresearch.descriptors import Descriptor
    from autoresearch.ledger import Entry

    sa = Descriptor([("simulated annealing", 6), ("block moves", 3)])
    ga = Descriptor([("genetic algorithm", 6), ("center-star alignment", 3)])
    e1, e2 = (Entry(id=i, parent_ids=[], mode="x", hypothesis="", status="kept") for i in (1, 2))
    rng = random.Random(0)
    # a lone strong solver has no untried pairing of its own parts
    assert ee.gap_directions([(e1, sa)], [sa], 5, rng) == []
    gaps = ee.gap_directions([(e1, sa), (e2, ga)], [sa, ga], 5, rng)
    assert len(gaps) == 2 and any("simulated annealing" in g and "center-star alignment" in g for g in gaps), gaps
    # a pairing that already exists anywhere in the described pool is not a gap
    tried = Descriptor([("simulated annealing", 6), ("center-star alignment", 3)])
    gaps = ee.gap_directions([(e1, sa), (e2, ga)], [sa, ga, tried], 5, rng)
    assert len(gaps) == 1 and "genetic algorithm" in gaps[0] and "block moves" in gaps[0], gaps
    print("  Gap-directions test passed!")


def test_describe_cache():
    print("Testing describe() through the on-disk cache, in parallel (fake claude)...")
    import types
    from concurrent.futures import ThreadPoolExecutor

    import autoresearch.descriptors as D
    import autoresearch.llm_calls as L

    calls = []

    def fake_run(cmd, input, **kw):
        assert kw.get("cwd") == tempfile.gettempdir(), kw.get("cwd")  # outside the repo: no project CLAUDE.md
        calls.append(1)
        out = {"total_cost_usd": 0.01, "structured_output": {
            "terms": [{"term": "hill climbing", "tier": 6}], "new_terms": [], "summary": "s"}}
        return types.SimpleNamespace(stdout=json.dumps(out), stderr="")

    real, L.subprocess = L.subprocess, types.SimpleNamespace(run=fake_run, TimeoutExpired=TimeoutError)
    tmp = Path(tempfile.mkdtemp(prefix="autoresearch-describe-"))
    try:
        cache = tmp / "descriptors.json"
        sources = [f"def solve(instance):\n    return '{c}'\n" for c in "abcdef"]
        with ThreadPoolExecutor(6) as pool:  # concurrent cache writes: the path that deadlocked
            futures = [pool.submit(D.describe, s, D.Vocabulary(), "p", cache_path=cache) for s in sources]
            descs = [f.result(timeout=10) for f in futures]
        assert all(d.core == "hill climbing" for d in descs)
        assert len(json.loads(cache.read_text())) == 6
        D.describe(sources[0], D.Vocabulary(), "p", cache_path=cache)  # cache hit: no new call
        assert len(calls) == 6, len(calls)
        usage = [json.loads(l) for l in (tmp / "llm_usage.jsonl").read_text().splitlines()]
        assert len(usage) == 6 and abs(sum(u["cost_usd"] for u in usage) - 0.06) < 1e-9, usage
    finally:
        L.subprocess = real
        shutil.rmtree(tmp, ignore_errors=True)
    print("  Describe-cache test passed!")


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


if __name__ == "__main__":
    random.seed(0)
    test_novelty_gate()
    test_import_guard()
    test_evaluator_boundary()
    test_agent_run()
    test_swarm()
    test_exploration_exploitation()
    test_lineage_parents()
    test_gap_directions()
    test_gate_describe_failure()
    test_describe_cache()
    test_viz_lineage()
    print("\nAll autoresearch tests passed!")
