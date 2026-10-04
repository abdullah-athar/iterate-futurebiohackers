"""Report a scripts/bench.sh (or bench_policies.sh) benchmark: per problem and arm, mean ± sd over seeds,
every seed listed, plus JSON/CSV exports and an optional HTML page.

Per run: best gain over the seed on the search objective, held-out gain of the selected program (chosen
without the holdout), spend by function (agent sessions from agents/*-agent.json, describe / plan /
diagnostic calls from llm_usage.jsonl; unreported costs counted as unknown, never 0), wall-clock, and,
when --target-gain is given (fix it before looking at results), the minutes and dollars to first reach it
("not reached" otherwise, never 0). Diversity runs add: families discovered / refined / in the final
top-k, exact repeats, semantic rejections, proximity flags, plan regenerations and drops, spend per
family, protected families that became competitive and the spend of protections that gained nothing.
Per arm: the held-out difference to the reference arm with its standard error (Welch); with 3-5 seeds
this is a first signal, not a significance test.

Usage: uv run python scripts/bench_report.py TAG [--ref A] [--target-gain 17] [--html]
                                                 [--calibration artifacts/calibration/calibration-*.json] [--out DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autoresearch.llm_calls import read_usage  # noqa: E402
from autoresearch_viz.load import load_run  # noqa: E402
from autoresearch_viz.metrics import summarize  # noqa: E402

# validated categorical palette (dataviz reference palette), fixed order, never cycled
ARM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#94a3b8"


def agent_spend(d: Path, ledger_usd: float) -> tuple[float, int]:
    """Agent-session spend from agents/*-agent.json (one per session, empty sessions included, in every arm),
    falling back to the ledger. Returns (known dollars, sessions whose cost was not reported)."""
    files = sorted((d / "agents").glob("*-agent.json"))
    if not files:
        return ledger_usd, 0
    usd, unknown = 0.0, 0
    for f in files:
        try:
            cost = json.loads(f.read_text()).get("total_cost_usd")
        except (json.JSONDecodeError, OSError):
            cost = None
        if cost is None:
            unknown += 1
        else:
            usd += float(cost)
    return usd, unknown


def side_table(d: Path) -> dict[int, dict]:
    p = d / "entry_signatures.json"
    return {int(k): v for k, v in json.loads(p.read_text()).items()} if p.exists() else {}


def family_of(e, by_id, side) -> str | None:
    if e.usage.get("family"):
        return e.usage["family"]
    if side.get(e.id, {}).get("family"):
        return side[e.id]["family"]
    nid = e.novelty.get("nearest_id") if e.status == "rejected_duplicate" else None
    return family_of(by_id[nid], by_id, side) if nid in by_id and nid != e.id else None


def timeline(run, aux: list[dict], seed_obj: float) -> list[dict]:
    """One point per proposal in ledger order: minutes since start, cumulative spend (agents + auxiliary calls
    made up to then), best objective so far and its gain over the seed. Auxiliary records without a
    timestamp (older runs) are counted at the end."""
    t0 = next((e["t"] for e in run.events if e.get("type") == "run_start"), run.entries[0].timestamp)
    aux_t = sorted((r.get("t", math.inf), r.get("cost_usd") or 0.0) for r in aux)
    pts, cum, best, i = [], 0.0, seed_obj, 0
    for e in run.proposals:
        cum += float(e.usage.get("cost_usd") or 0.0)
        while i < len(aux_t) and aux_t[i][0] <= e.timestamp:
            cum += aux_t[i][1]
            i += 1
        if e.scored and e.confirmed is not False and e.status != "rejected_duplicate" and e.objective < best:
            best = e.objective
        pts.append({"min": (e.timestamp - t0) / 60, "usd": cum, "best": best,
                    "gain": 100 * (seed_obj - best) / seed_obj if seed_obj else math.nan})
    tail = sum(c for t, c in aux_t[i:])
    if pts and tail:
        pts[-1]["usd"] += tail
    return pts


def run_row(d: Path, target_gain: float | None, top_k: int = 10) -> dict:
    meta = json.loads((d / "bench.json").read_text())
    run = load_run(d)
    s = summarize(run)
    hold = json.loads((d / "holdout.json").read_text()) if (d / "holdout.json").exists() else {}
    hs, hb = hold.get("seed", {}).get("score"), hold.get("best", {}).get("score")
    agent_usd, agent_unknown = agent_spend(d, s.cost_usd)
    aux = read_usage(d)
    by_fn = Counter()
    for r in aux:
        if r.get("cost_usd") is not None:
            by_fn[r.get("kind", "describe")] += float(r["cost_usd"])
    aux_usd = sum(by_fn.values())
    ts = {e["type"]: e["t"] for e in run.events if e.get("type") in ("run_start", "run_end")}
    seed_obj = run.seed.objective if run.seed else math.nan
    pts = timeline(run, aux, seed_obj)
    reach = next((p for p in pts if p["gain"] >= target_gain), None) if target_gain is not None else None
    by_id = {e.id: e for e in run.entries}
    side = side_table(d)
    fams = {e.id: family_of(e, by_id, side) for e in run.entries}
    scored = [e for e in run.proposals if e.scored and e.confirmed is not False and e.status != "rejected_duplicate"]
    valid_by_fam = Counter(fams[e.id] for e in scored if fams[e.id])
    top = sorted([e for e in run.entries if e.scored and e.confirmed is not False and e.status != "rejected_duplicate"],
                 key=lambda e: (e.objective, e.id))[:top_k]
    fam_cost, fam_tries = Counter(), Counter()
    for e in run.proposals:
        f = fams[e.id] or ("no code" if e.status in ("no_output", "rejected_guard") else "not described")
        fam_cost[f] += float(e.usage.get("cost_usd") or 0.0)
        fam_tries[f] += 1
    plans = [e for e in run.events if e.get("type") == "plan"]
    fam_events = [e for e in run.events if e.get("type") == "families"]
    prots = (fam_events[-1].get("protections") if fam_events else None) or []
    grace_cost = Counter()
    for e in run.proposals:
        if e.usage.get("grace_family"):
            grace_cost[e.usage["grace_family"]] += float(e.usage.get("cost_usd") or 0.0)
    top_fams = {fams[e.id] for e in top if fams[e.id]}
    protected = {p["family"]: p for p in prots}
    return {**meta,
            "gain": s.gain_vs_seed_pct,
            "held": 100 * (hs - hb) / hs if hs and hb is not None else math.nan,
            "agent_usd": agent_usd, "describe_usd": aux_usd, "usd": agent_usd + aux_usd,
            "usd_by_function": {"agent": round(agent_usd, 4), **{k: round(v, 4) for k, v in by_fn.items()}},
            "unknown_cost": agent_unknown + sum(1 for r in aux if r.get("cost_usd") is None),
            "minutes": (ts["run_end"] - ts["run_start"]) / 60 if len(ts) == 2 else math.nan,
            "evaluated": sum(1 for e in run.proposals if e.evals), "proposals": len(run.proposals),
            "gated": sum(1 for e in run.proposals if e.note.startswith("not selected by max-min")),
            "exact_repeats": sum(1 for e in run.proposals if e.status == "rejected_duplicate"
                                 and not e.note.startswith(("semantic", "not selected by max-min"))),
            "semantic_rejects": sum(1 for e in run.proposals if e.usage.get("distance_decision") == "semantic_reject"),
            "proximity_flags": sum(1 for e in run.proposals if e.usage.get("distance_decision") == "proximity_flagged"),
            "plan_regenerations": sum(max(0, p["attempts"] - 1) for p in plans),
            "plans_dropped": sum(1 for p in plans if p["status"] == "dropped"),
            "families_discovered": len(valid_by_fam), "families_refined": sum(1 for c in valid_by_fam.values() if c >= 2),
            "families_in_top": len(top_fams), "described": bool(side or any(e.usage.get("signature") for e in run.proposals)),
            "cost_by_family": dict(fam_cost.most_common()), "tries_by_family": dict(fam_tries.most_common()),
            "protected": sorted(protected),
            "protected_competitive": sorted(f for f in protected if f in top_fams),
            "protection_usd_without_gain": round(sum(grace_cost[f] for f, p in protected.items()
                                                     if not p.get("improvements")), 4),
            "target_minutes": None if reach is None else reach["min"], "target_usd": None if reach is None else reach["usd"],
            "timeline": pts,
            "entropy": [{"gen": e["gen"], "top_H": e["top"]["H"], "top_norm": e["top"]["normalized"],
                         "recent_H": e["recent"]["H"], "families": e["families"]} for e in fam_events]}


def ms(xs: list[float], digits: int = 2) -> str:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    return f"{st.mean(xs):.{digits}f} ± {st.stdev(xs):.{digits}f}" if len(xs) > 1 else f"{xs[0]:.{digits}f}"


def welch(a: list[float], b: list[float]) -> str:
    a, b = [x for x in a if not math.isnan(x)], [x for x in b if not math.isnan(x)]
    if len(a) < 2 or len(b) < 2:
        return "n/a"
    diff = st.mean(a) - st.mean(b)
    se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
    return f"{diff:+.2f} (se {se:.2f})"


def reached(g: list[dict], key: str) -> str:
    vals = [r[key] for r in g]
    hit = [v for v in vals if v is not None]
    if not hit:
        return "not reached"
    return f"{ms(hit, 1 if key == 'target_minutes' else 2)} ({len(hit)}/{len(vals)} reached)"


def markdown(rows, groups, args) -> str:
    synthetic = any(r.get("synthetic") for r in rows)
    out = [f"# Benchmark `{args.tag}`", "",
           *(["**SYNTHETIC RUNS** (scripts/offline_world.py): a functional check of the machinery, not measurements.", ""]
             if synthetic else []),
           f"Model {rows[0]['model']}, {rows[0]['agents']} agents per generation. Gains are % below the seed's score "
           "(higher is better); held-out = instances the search never saw, scored once at the end on the program "
           "selected by the search objective. With 3-5 seeds these are first signals, not significance tests.", ""]
    for problem in sorted({p for p, _ in groups}):
        arms = sorted((a for p, a in groups if p == problem), key=lambda a: (a != args.ref, a))
        ref = [r["held"] for r in groups.get((problem, args.ref), [])]
        out += [f"## {problem}", "",
                f"| arm | commit | flags | n | gain | held-out gain | held-out vs {args.ref} | total $ | minutes "
                "| evaluated / proposals | target: minutes | target: $ |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            g = groups[(problem, a)]
            col = lambda k: [r[k] for r in g]  # noqa: E731
            vs = "—" if a == args.ref else welch(col("held"), ref)
            out.append(f"| {a} | {g[0]['sha']} | `{g[0].get('flags') or '-'}` | {len(g)} | {ms(col('gain'))} | "
                       f"{ms(col('held'))} | {vs} | {ms(col('usd'))} | {ms(col('minutes'), 1)} | "
                       f"{st.mean(col('evaluated')):.1f} / {st.mean(col('proposals')):.1f} | "
                       f"{reached(g, 'target_minutes') if args.target_gain is not None else '-'} | "
                       f"{reached(g, 'target_usd') if args.target_gain is not None else '-'} |")
        out += ["", "| arm | families discovered / refined / in top | exact repeats | semantic rejects | proximity flags "
                "| plan regen / dropped | gate-dropped (Johann) | protected -> competitive | $ on protections w/o gain "
                "| $ by function |", "|---|---|---|---|---|---|---|---|---|---|"]
        for a in arms:
            g = groups[(problem, a)]
            col = lambda k: [r[k] for r in g]  # noqa: E731
            fams = (f"{st.mean(col('families_discovered')):.1f} / {st.mean(col('families_refined')):.1f} / "
                    f"{st.mean(col('families_in_top')):.1f}") if any(r["described"] for r in g) else "not described"
            fn = Counter()
            for r in g:
                fn.update(r["usd_by_function"])
            prot = sum(len(r["protected"]) for r in g)
            comp = sum(len(r["protected_competitive"]) for r in g)
            out.append(f"| {a} | {fams} | {st.mean(col('exact_repeats')):.1f} | {st.mean(col('semantic_rejects')):.1f} | "
                       f"{st.mean(col('proximity_flags')):.1f} | {st.mean(col('plan_regenerations')):.1f} / "
                       f"{st.mean(col('plans_dropped')):.1f} | {st.mean(col('gated')):.1f} | {prot} -> {comp} | "
                       f"{sum(col('protection_usd_without_gain')):.2f} | "
                       + ", ".join(f"{k} {v / len(g):.2f}" for k, v in fn.most_common()) + " |")
        out.append("")
    out += ["## Runs (every seed)", "", "| run | gain | held-out | $ | unknown costs | minutes | evaluated | "
            "families | target min / $ |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tgt = ("not reached" if r["target_minutes"] is None else f"{r['target_minutes']:.1f} / {r['target_usd']:.2f}") \
            if args.target_gain is not None else "-"
        out.append(f"| {r['name']} | {r['gain']:.2f} | {r['held']:.2f} | {r['usd']:.2f} | {r['unknown_cost']} | "
                   f"{r['minutes']:.1f} | {r['evaluated']}/{r['proposals']} | "
                   f"{r['families_discovered'] if r['described'] else '-'} | {tgt} |")
    return "\n".join(out) + "\n"


# ----- HTML ----------------------------------------------------------------------------------------------

def html_page(rows, groups, args, calibration: dict | None) -> str:
    from autoresearch_viz.charts import RefLine, Series, stacked_hbars, step_chart
    from autoresearch_viz.html import CSS, JS, _legend

    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != args.ref, a))
    color = {a: (ARM_COLORS[i] if i < len(ARM_COLORS) else OTHER) for i, a in enumerate(arms)}
    legend = _legend([(a, color[a]) for a in arms])
    synthetic = any(r.get("synthetic") for r in rows)
    parts = [f"<h1>Benchmark {escape(args.tag)}</h1>"
             + ("<p style='background:#fff7ed;border:1px solid #fdba74;border-radius:8px;padding:10px'><b>Synthetic runs</b>"
                " (scripts/offline_world.py): a functional check of the machinery, not measurements of the benchmark.</p>"
                if synthetic else "")
             + "<p class='muted'>Each line is one run (one seed); color = arm. "
             "The best score kept so far is drawn, not every candidate. Held-out scores are measured once, at the end, "
             "and are in the tables only.</p>"]
    for problem in sorted({p for p, _ in groups}):
        runs = [r for r in rows if r["problem"] == problem]
        refs = [RefLine(args.target_gain, f"target {args.target_gain:g}%")] if args.target_gain is not None else []
        by_cost = [Series(f"{r['arm']} s{r['seed']}", color[r["arm"]], [(0.0, 0.0)] + [(p["usd"], p["gain"]) for p in r["timeline"]])
                   for r in runs]
        by_time = [Series(f"{r['arm']} s{r['seed']}", color[r["arm"]], [(0.0, 0.0)] + [(p["min"], p["gain"]) for p in r["timeline"]])
                   for r in runs]
        parts += [f"<div class='card'><h2>{escape(problem)}: best gain vs cumulative spend</h2>{legend}",
                  step_chart(by_cost, refs=refs, x_label="cumulative spend (agents + describe/plan calls)", y_label="best gain over the seed (%)", x_unit="$"),
                  "</div>", f"<div class='card'><h2>{escape(problem)}: best gain vs wall-clock</h2>{legend}",
                  step_chart(by_time, refs=refs, x_label="minutes since start", y_label="best gain over the seed (%)"), "</div>"]
        # tries and spend per family, per arm (summed over seeds); top 7 families + Other
        tot = Counter()
        for r in runs:
            tot.update(r["tries_by_family"])
        fams = [f for f, _ in tot.most_common(7)]
        fcol = {f: ARM_COLORS[i] for i, f in enumerate(fams)}
        for key, title, unit in (("tries_by_family", "tries per family", "tries"),
                                 ("cost_by_family", "agent spend per family", "$")):
            bars = []
            for a in arms:
                agg = Counter()
                for r in runs:
                    if r["arm"] == a:
                        agg.update(r[key])
                segs = [(f, fcol[f], float(agg.get(f, 0))) for f in fams]
                segs.append(("Other", OTHER, float(sum(v for f, v in agg.items() if f not in fcol))))
                bars.append((a, segs))
            parts += [f"<div class='card'><h2>{escape(problem)}: {title} (all seeds)</h2>",
                      _legend([(f, fcol[f]) for f in fams] + [("Other", OTHER)]),
                      stacked_hbars([(a, [(f, c, round(v, 2)) for f, c, v in segs]) for a, segs in bars], as_share=True,
                                    unit=unit), "</div>"]
        ent = [Series(f"{r['arm']} s{r['seed']} top-k", color[r["arm"]], [(e["gen"], e["top_H"]) for e in r["entropy"]])
               for r in runs if r["entropy"]]
        if ent:
            parts += [f"<div class='card'><h2>{escape(problem)}: family entropy of the top-k per generation</h2>{legend}",
                      step_chart(ent, refs=[], x_label="generation", y_label="H (nats)"), "</div>"]
        shown = set()
        for r in sorted((r for r in runs if r["described"]), key=lambda r: r["seed"]):
            if r["arm"] not in shown:  # one card per arm (its lowest seed); every run is in the JSON export
                shown.add(r["arm"])
                parts.append(family_progress_card(r, fcol))
    if calibration:
        parts.append(calibration_card(calibration))
    parts.append(decision_log(rows))
    tables = markdown(rows, groups, args)
    parts.append("<div class='card'><h2>Tables</h2><pre style='white-space:pre-wrap;font-size:12px'>"
                 + escape(tables) + "</pre></div>")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>Benchmark {escape(args.tag)}</title><style>{CSS}"
            "</style></head><body><main class='wrap' style='max-width:1000px;margin:0 auto;padding:24px'>"
            + "".join(parts) + f"</main><script>{JS}</script></body></html>")


def family_progress_card(r: dict, fcol: dict) -> str:
    """Family records of one run. `fcol` is the page-wide family -> color map (color follows the family)."""
    from autoresearch_viz.charts import Series, step_chart
    from autoresearch_viz.html import _legend
    run = load_run(Path(r["path"]))
    by_id = {e.id: e for e in run.entries}
    side = side_table(Path(r["path"]))
    series, best, grace = {}, {}, defaultdict(list)
    for i, e in enumerate(run.proposals, start=1):
        f = family_of(e, by_id, side)
        if f and e.scored and e.status != "rejected_duplicate" and (f not in best or e.objective < best[f]):
            best[f] = e.objective
            series.setdefault(f, []).append((float(i), e.objective))
        g = e.usage.get("grace_family")
        if g:  # drawn at the family's record at that moment (its new record if this try set one)
            y = best.get(g)
            if y is not None:
                grace[g].append((float(i), y, f"#{e.id} funded try for {g} ({e.usage.get('override', '')})"))
    fams = sorted(series, key=lambda f: best[f])
    ss = [Series(f, fcol.get(f, OTHER), pts, markers=[(x, y, f"{f} record {y:g}") for x, y in pts] + grace.get(f, []))
          for f, pts in ((f, series[f]) for f in fams)]
    return (f"<div class='card'><h2>Family records: {escape(r['name'])}</h2>"
            + _legend([(s.label, s.color) for s in ss]) + "<p class='muted'>Best objective per family (lower is "
            "better) by try index; dots = new records and funded grace tries (start, prolongation, end of protection)."
            "</p>" + step_chart(ss, refs=[], x_label="try index", y_label="family record (objective)") + "</div>")


def calibration_card(cal: dict) -> str:
    rows = []
    for label, m in cal.get("metrics", {}).items():
        rel = defaultdict(list)
        for p in cal.get("pairs_detail", []):
            v = (p.get("d") or {}).get(label)
            if v is not None:
                rel[p["relation"]].append(v)
        w, h, ml = 900, 30 * len(rel) + 40, 170
        svg = [f"<svg class='chart' viewBox='0 0 {w} {h}' width='{w}' height='{h}'>"]
        for i, (name, vals) in enumerate(sorted(rel.items())):
            y = 20 + 30 * i
            svg.append(f"<text class='tick' x='{ml - 8}' y='{y + 4}' text-anchor='end'>{escape(name)} ({len(vals)})</text>")
            svg.append(f"<line class='grid' x1='{ml}' x2='{w - 20}' y1='{y}' y2='{y}'/>")
            for v in vals:
                svg.append(f"<circle cx='{ml + (w - ml - 20) * v:.1f}' cy='{y}' r='4' fill='{ARM_COLORS[0]}' "
                           f"fill-opacity='0.55' stroke='#fff' stroke-width='1'><title>{escape(name)}: {v:.3f}</title></circle>")
        thr = m.get("suggested_repeat_threshold")
        if thr is not None:
            x = ml + (w - ml - 20) * thr
            svg.append(f"<line x1='{x:.1f}' x2='{x:.1f}' y1='5' y2='{h - 15}' stroke='#64748b' stroke-dasharray='6 4'/>"
                       f"<text class='tick' x='{x + 4:.1f}' y='{h - 4}'>suggested threshold {thr}</text>")
        svg.append(f"<text class='tick' x='{ml}' y='{h - 4}'>0</text><text class='tick' x='{w - 26}' y='{h - 4}'>1</text></svg>")
        rows.append(f"<h3>{escape(label)}</h3>" + "".join(svg) + f"<p class='muted'>{escape(m.get('note', ''))}</p>")
    return "<div class='card'><h2>Distance calibration</h2>" + "".join(rows) + "</div>"


def decision_log(rows: list[dict], limit: int = 300) -> str:
    lines = []
    for r in rows:
        run = load_run(Path(r["path"]))
        for e in run.events:
            t = e.get("type")
            if t == "schedule":
                for o in e.get("overrides", []):
                    if "worker" in o:
                        lines.append((r["name"], e["gen"], "override", f"w{o['worker']} {o['from']} -> {o['to']}: "
                                      f"{o['reason']} {o.get('family') or ''} parent #{o.get('parent', '')}"))
                ctl = e.get("controller")
                if ctl and ctl.get("changed"):
                    lines.append((r["name"], e["gen"], "controller", f"{'ON' if ctl['active'] else 'off'}: {ctl['why']}"))
            elif t == "plan" and e.get("status") != "accepted":
                lines.append((r["name"], e["gen"], "plan", f"w{e['worker']} {e['status']} after {e['attempts']}"))
            elif t == "mode_drift":
                lines.append((r["name"], e["gen"], "mode drift", f"w{e['worker']} {e['from_family']} -> {e['to_family']}"))
        for e in run.proposals:
            if e.usage.get("distance_decision") in ("semantic_reject", "proximity_flagged", "combination_repeat"):
                lines.append((r["name"], e.generation, e.usage["distance_decision"], f"#{e.id} [{e.mode}] {e.note[:120]}"))
    body = "".join(f"<tr><td>{escape(str(a))}</td><td>{b}</td><td>{escape(c)}</td><td>{escape(d)}</td></tr>"
                   for a, b, c, d in lines[:limit])
    more = f"<p class='muted'>{len(lines) - limit} more in the events files.</p>" if len(lines) > limit else ""
    return ("<div class='card'><h2>Decision log</h2><table><tr><th>run</th><th>gen</th><th>kind</th><th>decision</th></tr>"
            + (body or "<tr><td colspan=4>no overrides, plan refusals or distance decisions</td></tr>") + "</table>"
            + more + "</div>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--ref", help="arm the held-out differences are taken against (default A, else initial, else first)")
    ap.add_argument("--target-gain", type=float, help="quality target (gain %% over the seed) fixed before the runs")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--html", action="store_true")
    ap.add_argument("--calibration", type=Path, help="calibration JSON from scripts/calibrate_distance.py")
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "artifacts" / "runs")
    ap.add_argument("--out", type=Path, help="output directory (default artifacts/bench)")
    args = ap.parse_args()
    dirs = sorted(d for d in args.runs_dir.glob(f"{args.tag}-*") if (d / "bench.json").exists())
    if not dirs:
        sys.exit(f"no finished benchmark runs for tag {args.tag!r} ({args.runs_dir}/{args.tag}-*/bench.json)")
    rows = []
    for d in dirs:
        r = run_row(d, args.target_gain, args.top_k)
        r["name"], r["path"] = d.name, str(d)
        rows.append(r)
    arms = {r["arm"] for r in rows}
    args.ref = args.ref or ("A" if "A" in arms else "initial" if "initial" in arms else sorted(arms)[0])
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["problem"], r["arm"])].append(r)
    text = markdown(rows, groups, args)
    print(text)
    out = args.out or ROOT / "artifacts" / "bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.md").write_text(text)
    (out / f"{args.tag}.json").write_text(json.dumps(rows, indent=1, default=str))
    flat = ["name", "arm", "sha", "flags", "problem", "seed", "model", "gain", "held", "usd", "agent_usd", "describe_usd",
            "unknown_cost", "minutes", "evaluated", "proposals", "exact_repeats", "semantic_rejects", "proximity_flags",
            "plan_regenerations", "plans_dropped", "families_discovered", "families_refined", "families_in_top",
            "protection_usd_without_gain", "target_minutes", "target_usd"]
    with (out / f"{args.tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    written = [f"{args.tag}.md", f"{args.tag}.json", f"{args.tag}.csv"]
    if args.html:
        cal = json.loads(args.calibration.read_text()) if args.calibration else None
        (out / f"{args.tag}.html").write_text(html_page(rows, groups, args, cal))
        written.append(f"{args.tag}.html")
    print(f"written {out}/: " + ", ".join(written))


if __name__ == "__main__":
    main()
