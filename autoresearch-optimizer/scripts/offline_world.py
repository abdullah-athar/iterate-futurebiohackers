"""Offline, synthetic world for the diversity scheduler: no LLM, no real evaluator.

Fake agents write tiny programs whose family and score are stated in comments; a fake describer reads the
family back; a fake evaluator returns the stated score. Families follow fixed score curves, so a scenario
like "this family starts worse but improves with refinements" can be checked through the real swarm loop,
scheduler and report. Runs made here carry "synthetic": true in bench.json: their numbers are a functional
check of the machinery, never a measurement of the real benchmark.

Usage: uv run python scripts/offline_world.py [--tag offline] [--seeds 0 1 2] [--generations 8] [--agents 4]
       then: uv run python scripts/bench_report.py offline --html
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau".split()


def sig(paradigm, mechs=(), details=(), raw="", unknown=()):
    from autoresearch.families import Signature
    return Signature(paradigm, raw, tuple(mechs), tuple(details), tuple(unknown))


def world_source(fam: str, score: float, uid: int, invalid: bool = False) -> str:
    rng = random.Random(uid)
    body = " ".join(rng.choice(WORDS) + str(rng.randint(0, 999)) for _ in range(60))
    return (f"# family={fam}\n# score={score:.1f}\n" + ("# invalid\n" if invalid else "")
            + f"NOISE = {body.split()!r}\n\n\ndef solve(instance):\n    return {uid!r}\n")


def fake_evaluate(problem_name, problem, source, split, budget_ms):
    from autoresearch.problem import EvalResult
    m = re.search(r"# score=([\d.]+)", source)
    score = float(m.group(1)) if m else 20000.0
    if "# invalid" in source:
        return EvalResult(split, 21000.0, 20000.0, error="crash")
    if split == "screen":
        return EvalResult(split, 0.0, 1.0)
    if split == "holdout":  # a fixed, score-dependent offset: the holdout is a different instance set
        return EvalResult(split, round(score * 1.02, 1), 20000.0)
    return EvalResult(split, score, 20000.0)


class World:
    """Fake agents: tune/fix_losers/merge stay in the parent's family (or the slot's target family); new_family
    takes the next family of `discoveries`. A refinement moves the parent's score by `step[family]` (+ noise)."""

    def __init__(self, first: dict, step: dict, discoveries: list[str], noise: float = 0.0, seed: int = 0,
                 cost: float = 0.01):
        self.first, self.step, self.discoveries, self.uid = first, step, list(discoveries), 0
        self.rng, self.noise, self.cost = random.Random(seed), noise, cost

    def propose_many(self, run, gen, assignments):
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
                score = pscore + self.step[fam] + self.rng.gauss(0, self.noise)
            self.uid += 1
            out.append({"assignment": a, "source": world_source(fam, max(score, 1.0), self.uid),
                        "hypothesis": f"{a.mode} {fam}",
                        "usage": {"model": "synthetic", "seconds": 1.0, "outcome": "ok", "cost_usd": self.cost}})
        return out


class patched_describer:  # noqa: N801
    """FamilyDescriber.describe_one reads the family from the source comment (no LLM)."""

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    def __enter__(self):
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


def fake_holdout():
    """Replace swarm.write_holdout (real evaluator) by the fake one; returns a restore function."""
    from autoresearch import swarm
    real = swarm.write_holdout

    def write(run, emit):
        entries = run.entries()
        seed, best = entries[0], run.archive(entries).global_best
        res = {label: (e.id, fake_evaluate(run.problem_name, run.problem, run.store.read_candidate(e), "holdout", 0))
               for label, e in (("seed", seed), ("best", best))}
        (run.store.root / "holdout.json").write_text(json.dumps(
            {label: {"id": i, **r.to_dict()} for label, (i, r) in res.items()}, indent=1))
        emit("holdout", seed=res["seed"][1].score, best=res["best"][1].score, best_id=best.id, baseline=20000.0)
    swarm.write_holdout = write
    return lambda: setattr(swarm, "write_holdout", real)


def world_run(root: Path, cfg, world: World, agents: int = 4, generations: int = 6, seed: int = 0):
    from autoresearch import swarm
    from autoresearch.ledger import RunStore
    from autoresearch.loop import ResearchRun, evaluate_candidate
    run = ResearchRun.create(RunStore(root), cfg, seed_source=world_source("local_search", 10000, 0),
                             evaluate=fake_evaluate)
    emit = swarm.Events(run, 0.0, log=lambda line: None)
    swarm.run_swarm(run, agents, 1, 1e9, world.propose_many,
                    lambda srcs: [evaluate_candidate("median_string", s, 1000, fake_evaluate) for s in srcs],
                    emit, seed=seed, max_generations=generations)
    events = [json.loads(l) for l in (run.store.root / "events.jsonl").read_text().splitlines()]
    return run, events


# a slow family that starts worse and improves fast, a hopeless one, and a leader that creeps
SCENARIO = {"first": {"alignment_consensus": 10600.0, "beam_search": 11000.0, "tabu_search": 10900.0},
            "step": {"local_search": -15.0, "alignment_consensus": -300.0, "beam_search": +50.0, "tabu_search": -20.0},
            "discoveries": ["beam_search", "alignment_consensus", "tabu_search"]}

CONFIGS = {  # same switches as scripts/bench_policies.sh
    "A": {},
    "B": {"distance_policy": "soft", "diverse_parent_selection": True},
    "C": {"entropy_controller": True, "family_grace": True},
    "D": {"distance_policy": "soft", "diverse_parent_selection": True, "entropy_controller": True, "family_grace": True},
}


def main() -> None:
    from autoresearch.loop import LoopConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="offline")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--agents", type=int, default=4)
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "artifacts" / "runs")
    args = ap.parse_args()
    restore = fake_holdout()
    try:
        with patched_describer():
            for seed in args.seeds:
                for arm, flags in CONFIGS.items():
                    d = args.runs_dir / f"{args.tag}-median_string-{arm}-s{seed}"
                    shutil.rmtree(d, ignore_errors=True)
                    cfg = LoopConfig(problem="median_string", descriptors=False, family_diagnostics=True,
                                     stagnation_generations=1, **flags)
                    world = World(SCENARIO["first"], SCENARIO["step"], SCENARIO["discoveries"], noise=40.0, seed=seed)
                    run, _ = world_run(d, cfg, world, args.agents, args.generations, seed)
                    (d / "bench.json").write_text(json.dumps({
                        "tag": args.tag, "arm": arm, "sha": "synthetic", "flags": " ".join(f"{k}={v}" for k, v in flags.items()),
                        "problem": "median_string", "seed": seed, "model": "synthetic (offline world)",
                        "agents": args.agents, "generations": args.generations, "synthetic": True}))
                    best = run.archive().global_best
                    print(f"{d.name}: best {best.objective:g} ({best.usage.get('family')}), {len(run.entries()) - 1} proposals")
    finally:
        restore()
    print(f"report: uv run python scripts/bench_report.py {args.tag} --html   (synthetic numbers, not measurements)")


if __name__ == "__main__":
    main()
