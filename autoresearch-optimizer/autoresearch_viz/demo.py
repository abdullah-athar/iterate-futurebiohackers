"""Generate clearly-labelled synthetic runs so the dashboard can be designed before real runs exist.

The instance names, set-median baselines and planted optima are the real medium-tier median-string
benchmark values; the proposals, hypotheses and scores are simulated. Every run gets
`"synthetic": true` in its config.json and a "(synthetic)" label.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

# (name, set-median baseline, planted/best-known) for the medium tier of median_string.
MEDIUM = [
    ("dna_promoter_25bp", 83, 73),
    ("dna_regulatory_40bp", 223, 180),
    ("dna_high_noise_30bp", 138, 118),
    ("dna_hamming_exact_30bp", 114, 84),
    ("protein_domain_20aa", 91, 68),
]
SMALL = [("dna_planted_small_fixed", 8, 8), ("dna_planted_small_indels", 40, 34), ("protein_short_motif", 10, 9)]
HOLDOUT = [("holdout_dna_30bp_s101", 131, 109), ("holdout_dna_35bp_s102", 171, 139), ("holdout_protein_22aa_s103", 97, 75)]

HYPOTHESES = {
    "tune": [
        "Raise local-search restarts from 3 to 8 within the time budget",
        "Use a larger neighbourhood: all substitutions + single indels",
        "Seed local search from the top-3 input strings instead of only the set-median",
        "Accept sideways moves for 50 iterations to escape plateaus",
        "Early-stop when no instance improves for 200 iterations and reinvest time in restarts",
        "Tighten time budget per restart; more restarts beat deeper ones on noisy DNA",
    ],
    "new_algorithm": [
        "Replace hill-climbing with column-wise majority vote over a progressive alignment (helps indel-heavy sets)",
        "Simulated annealing with Levenshtein-aware proposal moves and geometric cooling",
        "Steiner-string DP approximation: iterative pairwise alignment to a consensus",
        "Beam search over prefixes guided by summed edit-distance lower bounds",
        "Genetic algorithm with alignment-aware crossover and small population",
    ],
    "merge": [
        "Merge: alignment consensus initialiser (good on indel sets) + annealing refinement (good on hamming sets)",
        "Merge: switch strategy on metric — majority vote for hamming, alignment+local search for levenshtein",
        "Merge: run both front members for half the budget each and return the better",
    ],
    "random": ["Random mutation of the current best", "Shuffle operator order", "Perturb constants", "Swap loop bounds"],
}


class Flavour:
    def __init__(
        self,
        label: str,
        description: str,
        *,
        modes: list[str],
        reflect: bool,
        novelty_gate: bool,
        pareto: bool,
        p_gain: float,
        p_regress: float,
        p_dup: float,
        p_fail: float,
        p_screen: float,
        prompt_tokens: tuple[int, int],
        completion_tokens: tuple[int, int],
    ) -> None:
        self.label, self.description = label, description
        self.modes, self.reflect, self.novelty_gate, self.pareto = modes, reflect, novelty_gate, pareto
        self.p_gain, self.p_regress, self.p_dup, self.p_fail, self.p_screen = p_gain, p_regress, p_dup, p_fail, p_screen
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens


FLAVOURS = [
    Flavour(
        "reflect+pareto+gate (synthetic)",
        "Reflect-then-edit, per-instance Pareto archive, novelty gate, UCB over prompt modes",
        modes=["tune", "new_algorithm", "merge"],
        reflect=True,
        novelty_gate=True,
        pareto=True,
        p_gain=0.30,
        p_regress=0.15,
        p_dup=0.18,
        p_fail=0.06,
        p_screen=0.10,
        prompt_tokens=(3200, 4800),
        completion_tokens=(900, 1900),
    ),
    Flavour(
        "reflect, no novelty gate (synthetic)",
        "Same loop but duplicates are evaluated instead of skipped",
        modes=["tune", "new_algorithm", "merge"],
        reflect=True,
        novelty_gate=False,
        pareto=True,
        p_gain=0.30,
        p_regress=0.15,
        p_dup=0.18,
        p_fail=0.06,
        p_screen=0.10,
        prompt_tokens=(3200, 4800),
        completion_tokens=(900, 1900),
    ),
    Flavour(
        "greedy hill-climb (synthetic)",
        "Karpathy-style: single global best, tune-only prompts, no reflection",
        modes=["tune"],
        reflect=False,
        novelty_gate=False,
        pareto=False,
        p_gain=0.22,
        p_regress=0.25,
        p_dup=0.22,
        p_fail=0.08,
        p_screen=0.12,
        prompt_tokens=(1800, 2600),
        completion_tokens=(700, 1500),
    ),
    Flavour(
        "random mutation (synthetic)",
        "No LLM reasoning: random edits to the current best",
        modes=["random"],
        reflect=False,
        novelty_gate=False,
        pareto=False,
        p_gain=0.10,
        p_regress=0.45,
        p_dup=0.05,
        p_fail=0.15,
        p_screen=0.20,
        prompt_tokens=(600, 900),
        completion_tokens=(200, 500),
    ),
]


def _eval_block(split: str, scores: dict[str, float], table: list[tuple[str, int, int]], rng: random.Random) -> dict:
    insts = []
    for name, base, best_known in table:
        insts.append(
            {
                "name": name,
                "score": scores[name],
                "baseline": base,
                "best_known": best_known,
                "valid": True,
                "error": "",
                "elapsed": round(rng.uniform(0.4, 2.5), 2),
            }
        )
    return {
        "split": split,
        "score": sum(scores.values()),
        "baseline": sum(b for _, b, _ in table),
        "instances": insts,
        "elapsed": round(sum(i["elapsed"] for i in insts), 2),
        "error": "",
    }


def simulate(flavour: Flavour, steps: int, seed: int) -> tuple[dict, list[dict]]:
    rng = random.Random(seed)
    t = time.time() - 3600 * 24
    base_med = {n: b for n, b, _ in MEDIUM}
    opt_med = {n: k for n, _, k in MEDIUM}
    base_small = {n: b for n, b, _ in SMALL}

    entries: list[dict] = []
    # Seed = set-median baseline solver.
    seed_scores = dict(base_med)
    best_scores = dict(seed_scores)  # per-instance best so far (Pareto front summary)
    global_best = sum(seed_scores.values())
    front: list[tuple[int, dict[str, float]]] = [(0, dict(seed_scores))]
    entries.append(
        {
            "id": 0,
            "parent_ids": [],
            "mode": "seed",
            "hypothesis": "Seed solver: set-median baseline",
            "status": "seed",
            "proposer": "seed",
            "objective": global_best,
            "improved_global": True,
            "improved_instances": list(base_med),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "elapsed": 6.1,
            "timestamp": t,
            "evals": {
                "screen": _eval_block("screen", dict(base_small), SMALL, rng),
                "validate": _eval_block("validate", seed_scores, MEDIUM, rng),
            },
            "novelty": {},
            "note": "",
        }
    )
    since_gain = 0
    i = 0
    evaluated = 0
    while evaluated < steps:  # `steps` counts evaluations; gated duplicates only cost proposal tokens
        i += 1
        plateau = since_gain >= 4
        modes = flavour.modes
        if plateau and len(modes) > 1:
            modes = [m for m in modes if m != "tune"]
        mode = rng.choice(modes)
        hyp = rng.choice(HYPOTHESES[mode])
        if not flavour.reflect and mode != "random":
            hyp = hyp.split(":")[0].split(" within")[0]
        parent_id, parent_scores = front[0][0], best_scores
        if flavour.pareto and len(front) > 1 and rng.random() < 0.35:
            parent_id, parent_scores = rng.choice(front)
        parents = [parent_id]
        if mode == "merge" and len(front) > 1:
            parents = sorted({front[0][0], rng.choice(front)[0]})
        pt = rng.randint(*flavour.prompt_tokens)
        ct = rng.randint(*flavour.completion_tokens)
        llm_s = rng.uniform(6, 20)
        t += llm_s
        entry = {
            "id": i,
            "parent_ids": parents,
            "mode": mode,
            "hypothesis": hyp,
            "status": "evaluated",
            "proposer": "api:anthropic" if flavour.reflect else "api:gemini",
            "objective": None,
            "improved_global": False,
            "improved_instances": [],
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "elapsed": round(llm_s, 2),
            "timestamp": t,
            "evals": {},
            "novelty": {},
            "note": "",
        }
        is_dup = i > 2 and rng.random() < flavour.p_dup
        if is_dup and flavour.novelty_gate:
            entry["status"] = "rejected_duplicate"
            entry["novelty"] = {"similarity": round(rng.uniform(0.95, 1.0), 3), "duplicate_of": rng.choice(entries)["id"]}
            entries.append(entry)
            continue
        evaluated += 1
        r = rng.random()
        if r < flavour.p_fail:
            entry["status"] = "failed"
            entry["objective"] = float("inf")
            entry["evals"]["screen"] = {
                "split": "screen",
                "score": float("inf"),
                "baseline": sum(base_small.values()),
                "instances": [],
                "elapsed": 0.3,
                "error": rng.choice(
                    [
                        "IndexError: string index out of range",
                        "TimeoutError: solve() exceeded 30s on dna_planted_small_indels",
                        "ValueError: output contains symbol not in alphabet",
                    ]
                ),
            }
            t += 2
            entries.append(entry)
            since_gain += 1
            continue
        screen_scores = {n: max(b - rng.randint(0, 3), 0) for n, b in base_small.items()}
        entry["evals"]["screen"] = _eval_block("screen", screen_scores, SMALL, rng)
        t += entry["evals"]["screen"]["elapsed"]
        if r < flavour.p_fail + flavour.p_screen:
            entry["status"] = "rejected_screen"
            entry["evals"]["screen"]["score"] = sum(screen_scores.values()) + rng.randint(2, 9)
            entries.append(entry)
            since_gain += 1
            continue
        new_scores = {}
        for name, cur in parent_scores.items():
            if is_dup:
                new_scores[name] = cur
                continue
            u = rng.random()
            gain_mult = 1.6 if mode == "new_algorithm" else (1.3 if mode == "merge" else 1.0)
            if u < flavour.p_gain * gain_mult:
                room = cur - opt_med[name]
                new_scores[name] = cur - min(room, rng.randint(1, max(1, int(room * 0.25) + 1)))
            elif u < flavour.p_gain * gain_mult + flavour.p_regress:
                new_scores[name] = cur + rng.randint(1, 8)
            else:
                new_scores[name] = cur
        entry["evals"]["validate"] = _eval_block("validate", new_scores, MEDIUM, rng)
        t += entry["evals"]["validate"]["elapsed"]
        total = sum(new_scores.values())
        entry["objective"] = total
        improved_inst = [n for n, s in new_scores.items() if s < best_scores[n]]
        entry["improved_instances"] = improved_inst
        if total < global_best:
            global_best = total
            entry["improved_global"] = True
            front = [(i, new_scores)] + [f for f in front if any(f[1][n] < new_scores[n] for n in new_scores)]
            since_gain = 0
        else:
            since_gain += 1
            if improved_inst and flavour.pareto:
                front.append((i, new_scores))
        if entry["improved_global"] or (improved_inst and flavour.pareto):
            entry["status"] = "kept"
            for n in improved_inst:
                best_scores[n] = new_scores[n]
        elif improved_inst:
            for n in improved_inst:
                best_scores[n] = new_scores[n]
        entries.append(entry)

    # Held-out evaluation of the final best (fresh-seed instances the loop never saw).
    best_entry = min((e for e in entries if e["objective"] is not None and e["objective"] != float("inf")), key=lambda e: e["objective"])
    gain = (sum(base_med.values()) - best_entry["objective"]) / sum(base_med.values())
    hold_scores = {}
    for name, base, known in HOLDOUT:
        g = max(0.0, gain * rng.uniform(0.6, 1.0))
        hold_scores[name] = max(known, round(base * (1 - g)))
    best_entry["evals"]["holdout"] = _eval_block("holdout", hold_scores, HOLDOUT, rng)

    config = {
        "label": flavour.label,
        "description": flavour.description,
        "problem": "median_string",
        "synthetic": True,
        "objective_split": "validate",
        "novelty_gate": flavour.novelty_gate,
        "pareto_archive": flavour.pareto,
        "reflect": flavour.reflect,
        "modes": flavour.modes,
    }
    return config, entries


def write_demo_runs(out_dir: Path, steps: int = 40, seed: int = 7) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, flavour in enumerate(FLAVOURS):
        name = flavour.label.replace(" (synthetic)", "").replace(" ", "_").replace("+", "-").replace(",", "")
        root = out_dir / f"synthetic_{name}"
        root.mkdir(parents=True, exist_ok=True)
        config, entries = simulate(flavour, steps, seed + k)
        (root / "config.json").write_text(json.dumps(config, indent=2))
        with (root / "ledger.jsonl").open("w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        paths.append(root)
    return paths
