"""Tests for the diversity extension (families, distances, scheduler) and the reference fixes it rests on.

The LLM is faked at the subprocess boundary (llm_calls.subprocess.run) or by passing descriptors directly;
evaluation is either the real local evaluator or a deterministic fake. Run: uv run python scripts/test_diversity.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from autoresearch.json_cache import JsonCache  # noqa: E402

DESCRIPTOR_OUT = {"terms": [{"term": "hill climbing", "tier": 6}], "new_terms": [], "summary": "s"}


class FakeClaude:
    """Stands in for subprocess.run inside llm_calls: answers every call with `answer(prompt)`."""

    def __init__(self, answer=lambda prompt: DESCRIPTOR_OUT, cost: float | None = 0.01):
        self.answer, self.cost, self.prompts = answer, cost, []
        self.TimeoutExpired = TimeoutError

    def run(self, cmd, input, **kw):
        assert kw.get("cwd") == tempfile.gettempdir(), kw.get("cwd")
        self.prompts.append(input)
        out = {"structured_output": self.answer(input), "usage": {"input_tokens": 100, "output_tokens": 20}}
        if self.cost is not None:
            out["total_cost_usd"] = self.cost
        return types.SimpleNamespace(stdout=json.dumps(out), stderr="")


class patched_claude:  # noqa: N801 - used as a context manager
    def __init__(self, fake: FakeClaude):
        self.fake = fake

    def __enter__(self):
        import autoresearch.llm_calls as L
        self.L, self.real = L, L.subprocess
        L.subprocess = self.fake
        return self.fake

    def __exit__(self, *exc):
        self.L.subprocess = self.real


def tmpdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"autoresearch-{prefix}-"))


# ----- step 1: reference fixes -------------------------------------------------------------------

def test_describe_miss_then_hit():
    import autoresearch.descriptors as D
    tmp = tmpdir("describe")
    try:
        cache = tmp / "descriptors.json"
        with patched_claude(FakeClaude()) as fake:
            a = D.describe("def solve(i):\n    return 'a'\n", D.Vocabulary(), "p", cache_path=cache)
            b = D.describe("def solve(i):\n    return 'a'\n", D.Vocabulary(), "p", cache_path=cache)
        assert a.core == b.core == "hill climbing" and len(fake.prompts) == 1  # miss, then hit
        usage = [json.loads(l) for l in (tmp / "llm_usage.jsonl").read_text().splitlines()]
        assert len(usage) == 1 and usage[0]["kind"] == "describe" and usage[0]["cost_usd"] == 0.01
        assert usage[0]["usage"] == {"input_tokens": 100, "output_tokens": 20}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  describe: miss then hit, no deadlock, spend logged")


def test_describe_cache_invalidation():
    import autoresearch.descriptors as D
    src = "def solve(i):\n    return 'a'\n"
    keys = {D.describe_key(src, "p", "sonnet"), D.describe_key(src, "p", "haiku"), D.describe_key(src, "q", "sonnet")}
    assert len(keys) == 3, keys  # model and problem text are part of the key
    renamed = "def solve(inst):\n    # a comment\n    return 'a'\n"
    assert D.describe_key(src, "p", "sonnet") == D.describe_key(renamed, "p", "sonnet")  # cosmetic edits hit
    print("  describe: cache key changes with model/problem, not with cosmetic edits")


def test_unknown_cost_is_not_zero():
    import autoresearch.descriptors as D
    from autoresearch.budget import run_spend
    from autoresearch.ledger import RunStore
    tmp = tmpdir("unknown-cost")
    try:
        store = RunStore(tmp / "run")
        store.create({})
        with patched_claude(FakeClaude(cost=None)):
            D.describe("def solve(i):\n    return 'b'\n", D.Vocabulary(), "p", cache_path=store.root / "descriptors.json")
        rec = json.loads((store.root / "llm_usage.jsonl").read_text())
        assert rec["cost_usd"] is None
        s = run_spend(store)
        assert s.unknown_cost == 1 and s.usd == 0.0 and s.calls == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  usage: an unreported cost is logged as unknown (None), not 0")


def _writer(path: str, worker: int, n: int) -> None:
    cache = JsonCache(Path(path))
    for i in range(n):
        cache.put(f"w{worker}-{i}", {"worker": worker, "i": i})


def test_cache_concurrent_processes_and_threads():
    from concurrent.futures import ThreadPoolExecutor
    tmp = tmpdir("cache")
    try:
        path = tmp / "cache.json"
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_writer, args=(str(path), w, 25)) for w in range(3)]
        for p in procs:
            p.start()
        with ThreadPoolExecutor(4) as pool:  # threads of this process write at the same time
            list(pool.map(lambda w: _writer(str(path), w, 25), range(3, 7)))
        for p in procs:
            p.join(60)
            assert p.exitcode == 0, p.exitcode
        data = json.loads(path.read_text())  # valid JSON: no partial write
        assert len(data) == 7 * 25, len(data)  # no lost update
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  cache: 3 processes + 4 threads, 175 writes, none lost, file always valid")


def test_budget_caps_with_calls_in_flight():
    from autoresearch.budget import RunBudget
    from autoresearch.ledger import Entry, RunStore
    tmp = tmpdir("budget")
    try:
        store = RunStore(tmp / "run")
        store.create({})
        store.append(Entry(id=0, parent_ids=[], mode="seed", hypothesis="", status="seed", proposer="seed"))
        for i in (1, 2):
            store.append(Entry(id=i, parent_ids=[0], mode="tune", hypothesis="", status="evaluated",
                               proposer="claude-code:x", generation=1, usage={"cost_usd": 0.40}))
        store.append(Entry(id=3, parent_ids=[0], mode="tune", hypothesis="", status="no_output",
                           proposer="claude-code:x", generation=1, usage={"cost_usd": 0.30}))
        b = RunBudget(store, max_usd=2.0)
        assert abs(b.spent().usd - 1.10) < 1e-9 and b.spent().calls == 3  # the empty session is counted
        ok, why = b.reserve_generation(2, 0.40)   # 1.10 + 0.80 <= 2.0
        assert ok, why
        assert b.reserve_aux("describe") is not None  # 1.10 + 0.80 in flight + 0.05 estimate
        assert b.reserve_aux("describe") is not None  # 2.00 exactly
        assert b.reserve_aux("describe") is None      # would exceed: refused while the others are in flight
        b.release_generation()
        ok, why = b.reserve_generation(3, 0.40)   # 1.10 + 1.20 > 2.0
        assert not ok and "cost cap" in why, why
        assert RunBudget(store, max_calls=3).reserve_generation(1, None)[0] is False  # call cap
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  budget: caps hold with generation and auxiliary calls in flight")


# ----- step 2-3: vocabulary, signatures, distance, entropy ---------------------------------------------

def test_vocabulary_synonyms():
    from autoresearch.families import Vocab, signature_from_canonical
    v = Vocab.load("median_string")
    assert v.canonical("Hill Climbing", "paradigm") == v.canonical("greedy local improvement", "paradigm") == "local_search"
    assert v.canonical("insertion moves", "mechanism") == v.canonical("deletion move", "mechanism") == "indel_moves"
    assert Vocab.load("median_string_long").version == v.version
    # a known paradigm written out under "other" lands on its id: rewording cannot found a new family
    s = signature_from_canonical({"paradigm": "other", "paradigm_other": "Annealing-based search",
                                  "mechanisms": ["substitution_moves"], "details": [], "summary": "x"}, v)
    assert s.family == "simulated_annealing" and not s.provisional
    p = signature_from_canonical({"paradigm": "other", "paradigm_other": "Quantum Walks", "mechanisms": ["bogus"],
                                  "details": [], "summary": "x"}, v)
    assert p.family == "provisional:quantum walk" and p.provisional and p.unknown == ("bogus",)
    print("  vocabulary: synonyms and spellings map to one id; unknown paradigms stay provisional")


def sig(paradigm, mechs=(), details=(), raw="", unknown=()):
    from autoresearch.families import Signature
    return Signature(paradigm, raw, tuple(mechs), tuple(details), tuple(unknown))


def test_distance_properties():
    import math
    import random

    from autoresearch.distance import WeightedJaccard, nearest
    from autoresearch.families import Vocab
    m = WeightedJaccard(Vocab.load("median_string").role_weights)
    a = sig("local_search", ["substitution_moves", "indel_moves"], ["move_ordering"])
    b = sig("alignment_consensus", ["alignment_to_center", "column_vote", "local_search_polish"])
    assert m.distance(a, a).value == 0.0
    assert m.distance(a, b).value == m.distance(b, a).value == 1.0  # nothing shared
    c = sig("local_search", ["substitution_moves", "indel_moves"], ["move_ordering", "deadline_fraction"])
    assert abs(m.distance(a, c).value - 1 / 14) < 1e-12  # one detail more: 13/14 shared
    rng = random.Random(0)
    v = Vocab.load("median_string")
    for _ in range(200):  # bounds, symmetry, no NaN
        x = sig(rng.choice(v.ids("paradigm")), rng.sample(v.ids("mechanism"), rng.randint(0, 4)))
        y = sig(rng.choice(v.ids("paradigm")), rng.sample(v.ids("mechanism"), rng.randint(0, 4)))
        d1, d2 = m.distance(x, y).value, m.distance(y, x).value
        assert 0.0 <= d1 <= 1.0 and d1 == d2 and not math.isnan(d1)
    for bad in (None, sig(None)):  # missing / empty: unavailable, never a duplicate
        r = m.distance(a, bad)
        assert not r.available and r.value is None
    res, rid = nearest(a, [(1, None), (2, b), (3, c)], m)
    assert rid == 3 and res.available and res.components["only_b"] == ["d:deadline_fraction"]
    print("  distance: in [0,1], symmetric, 0 for identical, unavailable for empty input, components reported")


def test_entropy_cases():
    import math

    from autoresearch.families import entropy
    e = entropy({}, 10)
    assert e.empty and e.H == 0.0 and e.n == 0
    one = entropy({"local_search": 7}, 10)
    assert one.H == 0.0 and one.families == 1 and one.normalized == 0.0 and one.effective == 1.0
    bal = entropy({"a": 2, "b": 2, "c": 2, "d": 2}, 4)
    assert abs(bal.H - math.log(4)) < 1e-12 and abs(bal.normalized - 1.0) < 1e-12 and abs(bal.effective - 4) < 1e-9
    assert entropy({"a": 3}, 1).normalized is None  # denominator log(1) = 0: undefined, not 0 or 1
    print("  entropy: empty set, one family and balanced families follow the documented conventions")


# ----- step 4: family grace -----------------------------------------------------------------------------

def make_entries(spec, sense="min"):
    """spec: [(id, family, objective or None, extra usage)] -> ledger entries (entry 0 is the seed)."""
    from autoresearch.ledger import Entry
    out = []
    for i, fam, obj, extra in spec:
        status = "seed" if i == 0 else ("evaluated" if obj is not None else "failed")
        out.append(Entry(id=i, parent_ids=[] if i == 0 else [0], mode="seed" if i == 0 else "tune", hypothesis="",
                         status=status, proposer="seed" if i == 0 else "x", generation=None if i == 0 else i,
                         objective=float("inf") if obj is None else obj, usage={"family": fam, **(extra or {})}))
    return out


def grace_cfg(**kw):
    from autoresearch.loop import LoopConfig
    return LoopConfig(family_grace=True, **{"grace_epsilon": 1.0, **kw})


def test_family_record_rule():
    from autoresearch.families import FamilyArchive
    from autoresearch.scheduler import grace_states
    # maximisation: 70 -> 50 -> 65 sets no new record after 70 (no extension); 50 -> 55 -> 58 sets two
    for scores, improvements in (([70, 50, 65], 0), ([50, 55, 58], 2)):
        spec = [(0, "local_search", 0, None)] + [(i + 1, "beam_search", s, {"grace_family": "beam_search"} if i else None)
                                                 for i, s in enumerate(scores)]
        p = grace_states(FamilyArchive(make_entries(spec), sense="max"), {"local_search"}, grace_cfg(), True)[0]
        assert p.improvements == improvements, (scores, p)
    # minimisation with an epsilon larger than the gains: 100 -> 99.5 -> 99.2 is noise, not progress
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None),
            (2, "beam_search", 99.5, {"grace_family": "beam_search"}), (3, "beam_search", 99.2, {"grace_family": "beam_search"})]
    p = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, grace_cfg(grace_epsilon=1.0), True)[0]
    assert p.improvements == 0
    print("  grace: gain is measured against the family record (70->50->65: none; 50->55->58: two)")


def test_grace_budget_cap_and_expiry():
    from autoresearch.families import FamilyArchive
    from autoresearch.scheduler import grace_states
    cfg = grace_cfg(grace_min_evaluations=2, grace_max_evaluations=6, grace_patience=1)
    # first valid program + two failed grace tries (invalid candidates count): 3 of min(6, 2+0+1)=3 -> expired
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None),
            (2, "beam_search", None, {"grace_family": "beam_search"}), (3, None, None, {"grace_family": "beam_search"})]
    p = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, cfg, True)[0]
    assert (p.status, p.used, p.budget) == ("expired", 3, 3), p
    # steady progress extends it, but never past grace_max
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None)] + [
        (i, "beam_search", 100 - 10 * i, {"grace_family": "beam_search"}) for i in range(2, 9)]
    p = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, cfg, True)[0]
    assert p.budget == 6 and p.status == "expired", p
    # patience 0 (ablation): the first refinement without progress ends it
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None), (2, "beam_search", 120, {"grace_family": "beam_search"})]
    p = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, grace_cfg(grace_patience=0), True)[0]
    assert (p.status, p.used, p.budget) == ("expired", 2, 2), p
    print("  grace: failed candidates use the budget; progress extends it up to the cap; patience is optional")


def test_grace_not_renewed_by_renaming():
    from autoresearch.families import FamilyArchive
    from autoresearch.scheduler import grace_states
    cfg = grace_cfg(grace_patience=0)
    # beam_search expired; a later program labelled with a synonym is the same family id: still expired, no reset
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None), (2, "beam_search", 110, {"grace_family": "beam_search"}),
            (3, "beam_search", 90, None)]
    states = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, cfg, True)
    assert [p.family for p in states] == ["beam_search"] and states[0].status == "expired"
    # the seed's family never gets the new-family budget; a provisional family is refused
    spec = [(0, "local_search", 0, None), (1, "local_search", 90, None), (2, "provisional:quantum walk", 95, None)]
    states = grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, cfg, True)
    assert [(p.family, p.status) for p in states] == [("provisional:quantum walk", "refused")]
    # no budget for the minimal refinement: not admitted
    spec = [(0, "local_search", 0, None), (1, "beam_search", 100, None)]
    assert grace_states(FamilyArchive(make_entries(spec)), {"local_search"}, cfg, False)[0].status == "refused"
    print("  grace: no renewed bonus by renaming or dormancy; provisional and unfundable families are refused")


def test_reservations_atomic():
    import threading

    from autoresearch.scheduler import Reservations
    res = Reservations()
    winners, barrier = [], threading.Barrier(8)

    def go(w):
        barrier.wait()
        if res.reserve(("tune", (5,), "local_search", "indel_moves"), w) is None:
            winners.append(w)
    threads = [threading.Thread(target=go, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(winners) == 1, winners
    res.release(("tune", (5,), "local_search", "indel_moves"), winners[0])
    assert res.reserve(("tune", (5,), "local_search", "indel_moves"), 99) is None  # released -> free again
    print("  reservations: 8 simultaneous tasks, exactly one gets the plan; release frees it")


# ----- synthetic world: scheduler end to end through run_swarm ------------------------------------------

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def world_source(fam: str, score: float, uid: int, invalid: bool = False) -> str:
    import random
    rng = random.Random(uid)
    body = " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(60))
    return (f"# family={fam}\n# score={score:.1f}\n" + ("# invalid\n" if invalid else "")
            + f"NOISE = {body.split()!r}\n\n\ndef solve(instance):\n    return {uid!r}\n")


def fake_evaluate(problem_name, problem, source, split, budget_ms):
    import re

    from autoresearch.problem import EvalResult
    m = re.search(r"# score=([\d.]+)", source)
    score = float(m.group(1)) if m else 20000.0
    if "# invalid" in source:
        return EvalResult(split, 21000.0, 20000.0, error="crash")
    if split == "screen":
        return EvalResult(split, 0.0, 1.0)
    return EvalResult(split, score, 20000.0)


class World:
    """Fake agents: tune/fix_losers/merge stay in the parent's family; new_family takes the next family of
    `discoveries`. Each family's score moves by `step[family]` per refinement of its parent."""

    def __init__(self, first: dict, step: dict, discoveries: list[str]):
        self.first, self.step, self.discoveries, self.uid = first, step, list(discoveries), 0

    def propose_many(self, run, gen, assignments):
        import re
        by_id = {e.id: e for e in run.entries()}
        out = []
        for a in assignments:
            parent = by_id[a.parent_ids[0]]
            src = run.store.read_candidate(parent)
            pfam = re.search(r"# family=(\S+)", src).group(1)
            pscore = float(re.search(r"# score=([\d.]+)", src).group(1))
            if a.mode == "new_family" and not a.meta.get("target_family") and self.discoveries:
                fam = self.discoveries.pop(0)
                score = self.first[fam]
            else:
                fam = a.meta.get("target_family") or pfam
                score = pscore + self.step[fam]
            self.uid += 1
            out.append({"assignment": a, "source": world_source(fam, score, self.uid), "hypothesis": f"{a.mode} {fam}",
                        "usage": {"model": "fake", "seconds": 1.0, "outcome": "ok", "cost_usd": 0.01}})
        return out


class patched_describer:  # noqa: N801
    """FamilyDescriber.describe_one reads the family from the source comment (no LLM)."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    def __enter__(self):
        import re

        from autoresearch import families
        self.cls, self.real = families.FamilyDescriber, families.FamilyDescriber.describe_one

        def fake(describer, source):
            self.calls += 1
            if self.fail:
                raise AssertionError("describe must not be called")
            m = re.search(r"# family=(\S+)", source)
            fam = m.group(1) if m else "local_search"
            return sig(fam, ["substitution_moves"] if fam == "local_search" else ["column_vote"])
        self.cls.describe_one = fake
        return self

    def __exit__(self, *exc):
        self.cls.describe_one = self.real


def world_run(tmp: Path, cfg, world: World, agents: int = 4, generations: int = 6, seed: int = 0):
    from autoresearch import swarm
    from autoresearch.ledger import RunStore
    from autoresearch.loop import ResearchRun, evaluate_candidate
    run = ResearchRun.create(RunStore(tmp / "run"), cfg, seed_source=world_source("local_search", 10000, 0),
                             evaluate=fake_evaluate)
    emit = swarm.Events(run, 0.0, log=lambda line: None)
    swarm.run_swarm(run, agents, 1, 1e9, world.propose_many,
                    lambda srcs: [evaluate_candidate("median_string", s, 1000, fake_evaluate) for s in srcs],
                    emit, seed=seed, max_generations=generations)
    run.write_holdout_disabled = True
    events = [json.loads(l) for l in (run.store.root / "events.jsonl").read_text().splitlines()]
    return run, events


def _no_holdout():
    """write_holdout evaluates with the real evaluator; the synthetic world has none."""
    from autoresearch import swarm
    real = swarm.write_holdout
    swarm.write_holdout = lambda run, emit: None
    return lambda: setattr(swarm, "write_holdout", real)


SLOW = {"first": {"alignment_consensus": 10600.0, "beam_search": 11000.0},
        "step": {"local_search": -5.0, "alignment_consensus": -400.0, "beam_search": +50.0}}


def test_world_slow_family_gets_its_refinements():
    from autoresearch.loop import LoopConfig
    restore = _no_holdout()
    try:
        results = {}
        for grace in (True, False):
            tmp = tmpdir("world")
            try:
                with patched_describer():
                    cfg = LoopConfig(problem="median_string", descriptors=False, family_grace=grace,
                                     family_diagnostics=not grace)
                    run, events = world_run(tmp, cfg, World(SLOW["first"], SLOW["step"],
                                                            ["alignment_consensus", "beam_search"]))
                entries = run.entries()
                best = run.archive(entries).global_best
                fam = lambda e: e.usage.get("family")  # noqa: E731
                results[grace] = {
                    "best_family": fam(best),
                    "ac_tries": sum(1 for e in entries if fam(e) == "alignment_consensus"),
                    "ac_grace": sum(1 for e in entries if e.usage.get("grace_family") == "alignment_consensus"),
                    "beam_grace": sum(1 for e in entries if e.usage.get("grace_family") == "beam_search"),
                    "events": events}
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        on, off = results[True], results[False]
        # protection: the slow family is refined until it overtakes the immediate leader
        assert on["ac_grace"] >= 2 and on["best_family"] == "alignment_consensus", on
        # without it the bandit keeps refining the leader and the slow family stays at its first try
        assert off["ac_tries"] == 1 and off["best_family"] == "local_search", off
        # the hopeless family loses its bonus after min (2) + patience (1) funded evaluations
        assert on["beam_grace"] == 2, on["beam_grace"]
        sched = [e for e in on["events"] if e["type"] == "schedule"]
        reasons = [o["reason"] for s in sched for o in s["overrides"] if "worker" in o]
        assert "grace:initial" in reasons and "grace:progress" in reasons, reasons
        last = sched[-1]["protections"]
        assert {p["family"]: p["status"] for p in last}["beam_search"] == "expired", last
        assert all(s["ucb_modes"] for s in sched)  # the bandit's own decision is kept next to the overrides
    finally:
        restore()
    print("  world: a family that is bad at first but improves gets its refinements and takes the lead;"
          " a hopeless one loses its bonus after min + patience")


def test_controller_moves_budget_on_concentration_and_stagnation():
    from autoresearch.loop import LoopConfig
    restore = _no_holdout()
    tmp = tmpdir("world-ctl")
    try:
        with patched_describer():
            cfg = LoopConfig(problem="median_string", descriptors=False, entropy_controller=True,
                             stagnation_generations=1, explore_boost_fraction=0.5)
            # every new family is worse and local_search stops improving: concentrated + stagnating
            world = World({"beam_search": 12000.0, "tabu_search": 12500.0},
                          {"local_search": +1.0, "beam_search": +1.0, "tabu_search": +1.0}, ["beam_search", "tabu_search"])
            run, events = world_run(tmp, cfg, world, agents=4, generations=5)
        sched = [e for e in events if e["type"] == "schedule"]
        active = [s for s in sched if s["controller"]["active"]]
        assert active, [s["controller"] for s in sched]
        moved = [o for s in active for o in s["overrides"] if "worker" in o]
        assert moved and all(o["reason"].startswith("controller:") for o in moved), moved
        assert all(len([o for o in s["overrides"] if "worker" in o]) <= s["cap"] for s in sched)  # bounded
        assert len(run.entries()) == 1 + 4 * 5  # budget moved, not added
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)
    print("  controller: concentrated top + stagnating best moves bounded slots to exploration, adds none")


def test_all_off_is_the_reference():
    from autoresearch.loop import LoopConfig
    from autoresearch.scheduler import needs_families
    assert not needs_families(LoopConfig())
    restore = _no_holdout()
    tmp = tmpdir("world-ref")
    try:
        with patched_describer(fail=True) as d:  # any describe call would fail the test
            run, events = world_run(tmp, LoopConfig(problem="median_string", descriptors=False),
                                    World(SLOW["first"], SLOW["step"], ["alignment_consensus"]), generations=3)
        assert d.calls == 0
        kinds = {e["type"] for e in events}
        assert not kinds & {"schedule", "families", "distance_gate"}, kinds
        assert all(not e.usage.get("ucb_mode") for e in run.entries())
    finally:
        restore()
        shutil.rmtree(tmp, ignore_errors=True)
    print("  reference: with every option off, no describe call, no schedule change, no extra events")


def test_distance_policy_per_mode():
    from autoresearch.loop import LoopConfig, ResearchRun
    from autoresearch.ledger import RunStore
    from autoresearch.novelty import NoveltyVerdict
    from autoresearch.scheduler import DiversityLayer
    from autoresearch.swarm import Assignment
    tmp = tmpdir("policy")
    try:
        with patched_describer():
            cfg = LoopConfig(problem="median_string", descriptors=False, distance_policy="strict",
                             archive_threshold=0.5, batch_threshold=0.5, drift_threshold_max=0.5)
            run = ResearchRun.create(RunStore(tmp / "run"), cfg, seed_source=world_source("local_search", 10000, 0),
                                     evaluate=fake_evaluate)
            layer = DiversityLayer(run)
            ok = (([], NoveltyVerdict("x", 0.1, None, False)))

            def res(w, mode, fam):
                return {"assignment": Assignment(w, mode, [0]), "source": world_source(fam, 9000, 100 + w),
                        "hypothesis": "", "usage": {}, "pre": ok}
            results = [res(0, "tune", "local_search"),           # identical tags to the parent: accepted
                       res(1, "fix_losers", "local_search"),     # never judged on descriptors
                       res(2, "new_family", "local_search"),     # same family as the archive: rejected (strict)
                       res(3, "new_family", "beam_search"),      # different: accepted
                       res(4, "new_family", "beam_search")]      # same as worker 3 in this batch: rejected
            layer.after_precheck(results, 1, None)
        d = [r["usage"].get("distance_decision") for r in results]
        assert d == ["accepted", "not_gated", "semantic_reject", "accepted", "semantic_reject"], d
        assert results[2]["pre"][1].is_duplicate and results[2]["usage"]["gate_dropped"]
        assert results[4]["usage"]["distance"]["against"] == "batch"
        assert not results[0]["pre"][1].is_duplicate and not results[1]["pre"][1].is_duplicate
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  policy: tune with identical tags and fix_losers pass; strict drops only close new_family proposals")


def test_rejections_count_in_bandit_and_cost():
    from autoresearch.budget import run_spend
    from autoresearch.ledger import Entry, RunStore
    from autoresearch.loop import LoopConfig, ResearchRun
    tmp = tmpdir("bandit")
    try:
        run = ResearchRun.create(RunStore(tmp / "run"), LoopConfig(problem="median_string"),
                                 seed_source=world_source("local_search", 10000, 0), evaluate=fake_evaluate)
        for i, st in enumerate(["rejected_duplicate", "rejected_guard", "no_output"], start=1):
            run.store.append(Entry(id=i, parent_ids=[0], mode="new_family", hypothesis="", status=st,
                                   proposer="claude-code:x", generation=1, usage={"cost_usd": 0.2}))
        entries = run.entries()
        scores = run.mode_scores(entries, run.archive(entries))
        assert scores["new_family"] != float("inf")  # three zero-gain tries, not "untried"
        assert abs(run_spend(run.store).usd - 0.6) < 1e-9
        run.config.bandit_counts_all_attempts = False
        assert run.mode_scores(entries, run.archive(entries))["new_family"] == float("inf")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  accounting: rejected and empty sessions are bandit tries and their cost is counted")


def test_plan_first_regenerates_and_hands_back():
    from autoresearch.loop import LoopConfig, ResearchRun
    from autoresearch.ledger import RunStore
    from autoresearch.scheduler import DiversityLayer
    from autoresearch.swarm import Assignment

    def answer(prompt):
        refused = "previous plan was refused" in prompt
        if "mode `new_family`" in prompt:  # always proposes the seed's own family
            return {"paradigm": "local_search", "mechanisms": ["substitution_moves"], "details": [], "summary": "again",
                    "change": {"kind": "paradigm", "target": "local_search", "description": "same thing"}}
        target = "block_moves" if refused else "indel_moves"
        return {"paradigm": "local_search", "mechanisms": ["substitution_moves", target], "details": [],
                "summary": f"add {target}", "change": {"kind": "mechanism", "target": target, "description": f"add {target}"}}
    tmp = tmpdir("plan")
    try:
        with patched_describer():
            cfg = LoopConfig(problem="median_string", descriptors=False, semantic_retry_before_codegen=True,
                             distance_policy="soft", archive_threshold=0.5, batch_threshold=0.2)
            run = ResearchRun.create(RunStore(tmp / "run"), cfg, seed_source=world_source("local_search", 10000, 0),
                                     evaluate=fake_evaluate)
            layer = DiversityLayer(run)
            events = []
            with patched_claude(FakeClaude(answer)) as fake:
                out = layer.plan([Assignment(0, "tune", [0]), Assignment(1, "tune", [0]), Assignment(2, "new_family", [0])],
                                 1, __import__("random").Random(0), lambda kind, **d: events.append({"type": kind, **d}))
        plans = {e["worker"]: e for e in events if e["type"] == "plan"}
        statuses = sorted((w, p["status"], p["attempts"]) for w, p in plans.items())
        assert [s[1] for s in statuses] == ["accepted", "accepted", "dropped"], statuses
        assert sorted(p["attempts"] for w, p in plans.items() if w in (0, 1)) == [1, 2]  # one regenerated once
        assert plans[2]["attempts"] == 3  # 1 + max_proposal_regenerations, then handed back
        assert [a.worker for a in out] == [0, 1]  # the dropped task's agent is not run
        targets = {a.meta["plan_change"]["target"] for a in out}
        assert targets == {"indel_moves", "block_moves"}, targets
        usage = [json.loads(l) for l in (run.store.root / "llm_usage.jsonl").read_text().splitlines()]
        assert sum(1 for u in usage if u["kind"] == "plan") == len(fake.prompts) == 6  # every plan call is paid
        assert any("too close" in p for p in fake.prompts)  # the feedback names the neighbour
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  plan-first: a repeated plan is regenerated with feedback; a persistent repeat is handed back, not run")


TESTS = [test_describe_miss_then_hit, test_describe_cache_invalidation, test_unknown_cost_is_not_zero,
         test_cache_concurrent_processes_and_threads, test_budget_caps_with_calls_in_flight,
         test_vocabulary_synonyms, test_distance_properties, test_entropy_cases, test_family_record_rule,
         test_grace_budget_cap_and_expiry, test_grace_not_renewed_by_renaming, test_reservations_atomic,
         test_world_slow_family_gets_its_refinements, test_controller_moves_budget_on_concentration_and_stagnation,
         test_all_off_is_the_reference, test_distance_policy_per_mode, test_rejections_count_in_bandit_and_cost,
         test_plan_first_regenerates_and_hands_back]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nAll {len(TESTS)} diversity tests passed!")
