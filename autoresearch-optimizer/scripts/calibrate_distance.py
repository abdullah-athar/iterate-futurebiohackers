"""Calibrate a signature distance before using it to reject anything.

Pairs (with their relation and the provenance of the label):
  redescription      the same file described twice (two independent passes)          certain (controlled)
  reformulation      identifiers renamed + reformatted                                certain (controlled)
  param_change       numeric constants changed (ints x2, floats x0.5)                 certain (controlled)
  same_family        two reference solvers of one curated family (other mechanisms)  curated by hand (review it)
  paradigm_change    two reference solvers of different curated families              curated by hand (review it)
  lineage_tune / improving_tune / lineage_new_family / merge_parent    from --runs ledgers (parent, child)
                     — a genealogy relation, weak as a label: a "tune" child may well have changed paradigm.
Sources: the seed, scripts/descriptor_fixtures and the classical solvers. Dev/test split by source program
(fixture, or lineage root for run pairs), so no source appears on both sides.

On the dev pairs it proposes a repeat threshold between the same-algorithm relations and the
paradigm changes (None when they overlap); on the test pairs it reports missed repeats, distinct programs
rejected at that threshold, and run tune children that improved the score yet sit below it (useful
improvements a filter would have blocked). It never claims a threshold transfers to another metric.

Costs LLM calls (two passes over every source and transformation). --dry-run prints the plan and the
number of calls without calling anything.

Usage: uv run python scripts/calibrate_distance.py [--out DIR] [--model sonnet] [--backends canonical existing]
                                                   [--runs RUN_DIR ...] [--max-run-pairs 30] [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.distance import get_metric  # noqa: E402
from autoresearch.families import FamilyDescriber, Vocab  # noqa: E402
from autoresearch.ledger import RunStore  # noqa: E402
from autoresearch.problem import get_problem  # noqa: E402

CURATED = {  # fixture -> family; labels by the author of this script, to be reviewed
    "beam_search": "beam_search", "center_star_consensus": "alignment_consensus", "editop_voting": "local_search",
    "genetic_algorithm": "population_search", "hill_climb": "local_search",
    "iterated_local_search": "iterated_local_search", "positional_vote": "positional_consensus",
    "set_median": "set_median_selection", "simulated_annealing": "simulated_annealing", "tabu_search": "tabu_search",
    "seed": "local_search", "solver_frequency_consensus": "positional_consensus",
    "solver_random_solver": "random_sampling", "solver_set_median": "set_median_selection",
}
SAME = ("redescription", "reformulation", "param_change")


def sources() -> dict[str, str]:
    out = {"seed": (ROOT / "autoresearch" / "seeds" / "median_string_seed.py").read_text()}
    for f in sorted((ROOT / "scripts" / "descriptor_fixtures").glob("*.py")):
        out[f.stem] = f.read_text()
    for name in ("frequency_consensus", "random_solver", "set_median"):
        out[f"solver_{name}"] = (ROOT / "median_string" / "solvers" / f"{name}.py").read_text()
    return out


class _Rename(ast.NodeTransformer):
    def __init__(self, keep: set[str]):
        self.keep, self.map = keep, {}

    def _n(self, name: str) -> str:
        return name if name in self.keep else self.map.setdefault(name, f"v{len(self.map)}_{name[:2]}")

    def visit_Name(self, node):
        node.id = self._n(node.id)
        return node

    def visit_arg(self, node):
        node.arg = self._n(node.arg)
        return node

    def visit_FunctionDef(self, node):
        if node.name != "solve":
            node.name = self._n(node.name)
        self.generic_visit(node)
        return node


def reformulate(src: str) -> str:
    tree = ast.parse(src)
    keep = set(dir(builtins)) | {"solve", "instance", "self"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep.update((a.asname or a.name).split(".")[0] for a in node.names)
        if isinstance(node, ast.ClassDef):
            keep.add(node.name)
        if isinstance(node, ast.Attribute):
            keep.add(node.attr)
    return "# reformulated: identifiers renamed\n" + ast.unparse(_Rename(keep).visit(tree)) + "\n"


class _Params(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return node
        if isinstance(node.value, int) and abs(node.value) >= 2:
            return ast.copy_location(ast.Constant(node.value * 2), node)
        if isinstance(node.value, float) and node.value > 0:
            return ast.copy_location(ast.Constant(node.value * 0.5), node)
        return node


def change_params(src: str) -> str:
    return ast.unparse(_Params().visit(ast.parse(src))) + "\n"


def group_split(group: str) -> str:
    return "dev" if int(hashlib.sha1(group.encode()).hexdigest(), 16) % 2 == 0 else "test"


def build_pairs(srcs: dict[str, str], runs: list[Path], max_run_pairs: int) -> tuple[list[dict], dict[str, str]]:
    """Pairs reference items by key; items[key] = source text (pass 1 and pass 2 describe every item)."""
    items, pairs = {}, []
    for name, src in srcs.items():
        items[name] = src
        pairs.append({"a": name, "b": name, "relation": "redescription", "provenance": "controlled", "group": name})
        for rel, fn in (("reformulation", reformulate), ("param_change", change_params)):
            try:
                items[f"{name}~{rel}"] = fn(src)
                pairs.append({"a": name, "b": f"{name}~{rel}", "relation": rel, "provenance": "controlled", "group": name})
            except SyntaxError:
                pass
    names = sorted(srcs)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if x in CURATED and y in CURATED:
                rel = "same_family" if CURATED[x] == CURATED[y] else "paradigm_change"
                pairs.append({"a": x, "b": y, "relation": rel, "provenance": "curated (review)", "group": min(x, y)})
    for run in runs:
        store = RunStore(run)
        entries = store.entries()
        by_id = {e.id: e for e in entries}
        n = 0
        for e in entries:
            if n >= max_run_pairs or not e.source_path or not e.parent_ids or e.mode == "seed" or e.parent_ids[0] not in by_id:
                continue
            p = by_id[e.parent_ids[0]]
            if not p.source_path:
                continue
            rel = {"tune": "improving_tune" if e.improved_global or e.improved_instances else "lineage_tune",
                   "new_family": "lineage_new_family", "merge": "merge_parent"}.get(e.mode)
            if not rel:
                continue
            ka, kb = f"{run.name}#{p.id}", f"{run.name}#{e.id}"
            items[ka], items[kb] = store.read_candidate(p), store.read_candidate(e)
            root = p.id
            while by_id[root].parent_ids and by_id[root].mode not in ("seed", "new_family") and by_id[root].parent_ids[0] in by_id:
                root = by_id[root].parent_ids[0]
            pairs.append({"a": ka, "b": kb, "relation": rel, "provenance": "lineage (weak)", "group": f"{run.name}#{root}",
                          "improved": bool(e.improved_global or e.improved_instances)})
            n += 1
    for p in pairs:
        p["split"] = group_split(p["group"])
    return pairs, items


def stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), 4), "median": round(st.median(xs), 4), "mean": round(st.mean(xs), 4),
            "max": round(max(xs), 4)}


def analyse(pairs: list[dict], metric_name: str) -> dict:
    out = {"relations": {}}
    for split in ("dev", "test"):
        for rel in sorted({p["relation"] for p in pairs}):
            ds = [p["d"][metric_name] for p in pairs if p["split"] == split and p["relation"] == rel]
            out["relations"][f"{split}:{rel}"] = stats(ds)
    dev_same = [p["d"][metric_name] for p in pairs if p["split"] == "dev" and p["relation"] in SAME and p["d"][metric_name] is not None]
    dev_diff = [p["d"][metric_name] for p in pairs if p["split"] == "dev" and p["relation"] == "paradigm_change"
                and p["d"][metric_name] is not None]
    thr = None
    if dev_same and dev_diff and min(dev_diff) > max(dev_same):
        thr = round((max(dev_same) + min(dev_diff)) / 2, 4)
    out["dev_max_same"] = max(dev_same) if dev_same else None
    out["dev_min_paradigm_change"] = min(dev_diff) if dev_diff else None
    out["suggested_repeat_threshold"] = thr
    out["note"] = ("dev same-algorithm and paradigm-change distances overlap: keep the semantic filter in observe mode"
                   if thr is None else "threshold placed midway between dev max(same) and dev min(paradigm change)")
    if thr is not None:
        test = [p for p in pairs if p["split"] == "test" and p["d"][metric_name] is not None]
        same = [p for p in test if p["relation"] in SAME]
        diff = [p for p in test if p["relation"] == "paradigm_change"]
        imp = [p for p in test if p["relation"] == "improving_tune"]
        out["test"] = {
            "missed_repeats": f"{sum(p['d'][metric_name] >= thr for p in same)}/{len(same)}",
            "distinct_rejected": f"{sum(p['d'][metric_name] < thr for p in diff)}/{len(diff)}",
            "useful_improvements_blocked": (f"{sum(p['d'][metric_name] < thr for p in imp)}/{len(imp)}" if imp
                                            else "unknown (no improving run pairs on the test side; pass --runs)"),
        }
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "calibration")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--backends", nargs="+", default=["canonical"], choices=["canonical", "existing"])
    ap.add_argument("--runs", nargs="*", type=Path, default=[])
    ap.add_argument("--max-run-pairs", type=int, default=30)
    ap.add_argument("--problem", default="median_string")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-usd", type=float, default=5.0, help="spend cap for all describe calls (refused calls = undescribed)")
    args = ap.parse_args(argv)
    pairs, items = build_pairs(sources(), args.runs, args.max_run_pairs)
    calls = 2 * len(items) * len(args.backends)
    print(f"{len(pairs)} pairs over {len(items)} programs; relations: "
          + ", ".join(f"{r}={sum(p['relation'] == r for p in pairs)}" for r in sorted({p['relation'] for p in pairs})))
    print(f"about {calls} describe calls ({len(args.backends)} backend(s) x 2 passes x {len(items)} programs)")
    if args.dry_run:
        return
    vocab = Vocab.load(args.problem)
    problem = " ".join(get_problem(args.problem).describe().strip().splitlines()[:4])
    stamp = time.strftime("%Y%m%d-%H%M%S")
    work = args.out / f"calibration-{stamp}"
    report = {"model": args.model, "vocab": vocab.version, "pairs": len(pairs), "programs": len(items), "metrics": {}}
    keys = sorted(items)
    from autoresearch.budget import CallBudget
    budget = CallBudget(args.max_usd)
    for backend in args.backends:
        passes = []
        for k in (1, 2):  # separate caches: pass 2 never hits pass 1
            d = FamilyDescriber(work / f"{backend}-pass{k}", problem, vocab, backend, args.model, budget=budget,
                                kind="describe:calibration", workers=8)
            sigs = d.describe_many([items[x] for x in keys])
            passes.append(dict(zip(keys, sigs)))
        metric_names = ["weighted_jaccard"] + (["minilm"] if backend == "existing" else [])
        for mname in metric_names:
            metric = get_metric(mname, vocab.role_weights)
            label = f"{backend}/{mname}"
            for p in pairs:
                r = metric.distance(passes[0][p["a"]], passes[1][p["b"]])
                p.setdefault("d", {})[label] = r.value
            report["metrics"][label] = {"version": metric.version, **analyse(pairs, label)}
        report.setdefault("undescribed", {})[backend] = sum(1 for x in keys if passes[0][x] is None or passes[1][x] is None)
    report["spend"] = {"usd": round(budget.spent_usd, 4), "calls": budget.calls, "refused": budget.refused,
                       "unknown_cost": budget.unknown, "cap_usd": args.max_usd}
    report["pairs_detail"] = pairs
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"calibration-{stamp}.json"
    path.write_text(json.dumps(report, indent=1))
    lines = [f"# Distance calibration ({stamp})", "", f"model {args.model}, vocabulary {vocab.version}, "
             f"{len(pairs)} pairs over {len(items)} programs; spend ${budget.spent_usd:.2f} over {budget.calls} calls "
             f"(cap ${args.max_usd:g}, {budget.refused} refused, {budget.unknown} with unknown cost)", ""]
    for label, m in report["metrics"].items():
        lines += [f"## {label}", "", "| split:relation | n | min | median | mean | max |", "|---|---|---|---|---|---|"]
        for rel, s in m["relations"].items():
            if s.get("n"):
                lines.append(f"| {rel} | {s['n']} | {s['min']} | {s['median']} | {s['mean']} | {s['max']} |")
        lines += ["", f"suggested repeat threshold: {m['suggested_repeat_threshold']} — {m['note']}"]
        if m.get("test"):
            lines += [f"test: missed repeats {m['test']['missed_repeats']}, distinct rejected "
                      f"{m['test']['distinct_rejected']}, useful improvements blocked {m['test']['useful_improvements_blocked']}"]
        lines.append("")
    (args.out / f"calibration-{stamp}.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"written {path}")


if __name__ == "__main__":
    main()
