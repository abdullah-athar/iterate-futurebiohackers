"""Render a one-page HTML comparison of swarm runs made with different models.

Quality (best objective, holdout gain) and efficiency (total cost, wall clock, gain per dollar) are
drawn as separate small multiples, then best-so-far against cumulative cost and wall clock.

Usage:
    uv run python scripts/compare_models.py Haiku=artifacts/runs/a Sonnet=artifacts/runs/b Opus=artifacts/runs/c \
        -o results/model-comparison.html
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

SERIES = ["--series-1", "--series-2", "--series-3"]  # fixed categorical order, one slot per run


@dataclass
class RunStats:
    label: str
    color: str
    seed: float
    best: float
    planted: float
    classical: tuple[str, float]
    holdout_seed: float
    holdout_best: float
    cost: float
    seconds: float
    tokens: int
    model: str
    curve: list[tuple[float, float, float]]  # (cumulative $, seconds since start, best so far)

    @property
    def gain_pct(self) -> float:
        return 100 * (self.seed - self.best) / self.seed

    @property
    def holdout_gain_pct(self) -> float:
        return 100 * (self.holdout_seed - self.holdout_best) / self.holdout_seed

    @property
    def gain_per_dollar(self) -> float:
        return (self.seed - self.best) / self.cost if self.cost else 0.0


def load(label: str, root: Path, color: str) -> RunStats:
    entries = [json.loads(l) for l in (root / "ledger.jsonl").read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in (root / "events.jsonl").read_text().splitlines() if l.strip()]
    seed_eval = entries[0]["evals"]["validate"]
    t0 = next(e["t"] for e in events if e["type"] == "run_start")
    t1 = next(e["t"] for e in events if e["type"] == "run_end")
    curve = [(0.0, 0.0, float(seed_eval["score"]))]
    curve += [(e["cost_usd"], e["t"] - t0, float(e["best"])) for e in events if e["type"] == "gen_end"]
    baselines = json.loads((root / "baselines.json").read_text())
    classical = min(((k, v["validate"]["score"]) for k, v in baselines.items() if k != "set_median"), key=lambda kv: kv[1])
    hold = json.loads((root / "holdout.json").read_text())
    proposals = entries[1:]
    model = next((e["proposer"].split(":", 1)[-1] for e in proposals if e.get("proposer")), "?")
    spend: dict[str, float] = {}  # exact model id = the one that cost the most across the agent sessions
    for f in (root / "agents").glob("*-agent.json"):
        try:
            usage = json.loads(f.read_text()).get("modelUsage") or {}
        except (json.JSONDecodeError, OSError):
            continue
        for mid, u in usage.items():
            spend[mid] = spend.get(mid, 0.0) + (u.get("costUSD") or 0.0)
    model = max(spend, key=spend.get) if spend else model
    return RunStats(
        label=label, color=color, seed=float(seed_eval["score"]), best=curve[-1][2],
        planted=float(sum(i["best_known"] for i in seed_eval["instances"])), classical=classical,
        holdout_seed=float(hold["seed"]["score"]), holdout_best=float(hold["best"]["score"]),
        cost=sum((e.get("usage") or {}).get("cost_usd") or 0.0 for e in proposals), seconds=t1 - t0,
        tokens=sum(e.get("prompt_tokens", 0) + e.get("completion_tokens", 0) for e in proposals), model=model,
        curve=curve,
    )


def fmt(v: float, kind: str) -> str:
    if kind == "pct":
        return f"{v:.1f}%"
    if kind == "usd":
        return f"${v:.2f}"
    if kind == "min":
        return f"{int(v // 60)} min {int(round(v % 60)):02d} s"
    return f"{v:,.0f}"


def bars(title: str, note: str, runs: list[RunStats], value, kind: str, higher_is_better: bool) -> str:
    """A small horizontal bar chart: one bar per run, from zero, the winner's label in bold."""
    vals = [value(r) for r in runs]
    best = max(vals) if higher_is_better else min(vals)
    W, label_w, row_h, top = 340, 64, 30, 8
    plot_w = W - label_w - 92
    vmax = max(vals) or 1
    rows = []
    for i, (r, v) in enumerate(zip(runs, vals)):
        y = top + i * row_h
        w = max(2.0, plot_w * v / vmax)
        win = v == best
        tip = f"{r.label}: {fmt(v, kind)}"
        rows.append(
            f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" class="{"lab win" if win else "lab"}">{escape(r.label)}</text>'
            f'<rect x="{label_w}" y="{y + 4}" width="{w:.1f}" height="16" rx="4" style="fill:var({r.color})" data-tip="{escape(tip)}"/>'
            f'<text x="{label_w + w + 6:.1f}" y="{y + 16}" class="{"val win" if win else "val"}">{fmt(v, kind)}{" ★" if win else ""}</text>'
        )
    h = top + len(runs) * row_h + 4
    return (f'<figure class="mini"><figcaption><b>{escape(title)}</b><span>{escape(note)}</span></figcaption>'
            f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{escape(title)}">'
            f'<line x1="{label_w}" x2="{label_w}" y1="{top}" y2="{h - 4}" class="axis"/>{"".join(rows)}</svg></figure>')


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    raw = span / n
    mag = 10 ** len(str(int(raw))) / 10 if raw >= 1 else 0.1
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = (lo // step + 1) * step
    out, v = [], start
    while v < hi:
        out.append(round(v, 6))
        v += step
    return out


def tradeoff(runs: list[RunStats], axis: str) -> str:
    """Best objective so far (lower is better) against cumulative cost or wall clock; steps at generation ends."""
    idx, xlab, kind = (0, "cumulative agent cost (Claude Code estimate)", "usd") if axis == "cost" else (1, "wall clock since the run started", "min")
    W, H, L, R, T, B = 760, 380, 70, 190, 18, 46
    xmax = max(p[idx] for r in runs for p in r.curve) * 1.05
    planted, classical, seed = runs[0].planted, runs[0].classical, runs[0].seed
    ylo = min(min(r.best for r in runs), planted) - 300
    yhi = max(seed, classical[1]) + 250
    X = lambda v: L + (W - L - R) * v / xmax
    Y = lambda v: T + (H - T - B) * (yhi - v) / (yhi - ylo)
    parts = []
    for tv in nice_ticks(ylo, yhi):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="grid"/>'
                     f'<text x="{L - 8}" y="{Y(tv) + 4:.1f}" text-anchor="end" class="tick">{tv:,.0f}</text>')
    for tv in nice_ticks(0, xmax, 6):
        lab = f"${tv:.2f}" if kind == "usd" else f"{tv / 60:.0f} min" if tv % 60 == 0 else f"{tv / 60:.1f} min"
        parts.append(f'<text x="{X(tv):.1f}" y="{H - B + 18}" text-anchor="middle" class="tick">{lab}</text>')
    parts.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" class="axis"/>')
    for v, name in ((seed, "seed solver"), (classical[1], f"best classical ({classical[0]})"),
                    (planted, "planted reference")):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="ref"/>'
                     f'<text x="{W - R + 6}" y="{Y(v) + 4:.1f}" class="refl">{escape(name)} {v:,.0f}</text>')
    # end labels, nudged apart so they never overlap
    ends = sorted(((Y(r.best), r) for r in runs), key=lambda t: t[0])
    placed, last = {}, -1e9
    for y, r in ends:
        y = max(y, last + 15)
        placed[r.label], last = y, y
    for r in runs:
        pts = r.curve
        d = f"M{X(pts[0][idx]):.1f},{Y(pts[0][2]):.1f}"
        for (a, b) in zip(pts, pts[1:]):
            d += f" H{X(b[idx]):.1f} V{Y(b[2]):.1f}"
        parts.append(f'<path d="{d}" class="line" style="stroke:var({r.color})"/>')
        for i, p in enumerate(pts[1:], 1):
            tip = f"{r.label}, after generation {i}: best {p[2]:,.0f} · {fmt(p[0], 'usd')} · {fmt(p[1], 'min')}"
            parts.append(f'<circle cx="{X(p[idx]):.1f}" cy="{Y(p[2]):.1f}" r="5" class="dot" style="fill:var({r.color})" data-tip="{escape(tip)}"/>')
        xe = X(pts[-1][idx])
        parts.append(f'<text x="{xe + 9:.1f}" y="{placed[r.label] + 4:.1f}" class="endl">{escape(r.label)} {r.best:,.0f}</text>')
    parts.append(f'<text x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle" class="axl">{escape(xlab)}</text>'
                 f'<text x="16" y="{(T + H - B) / 2}" text-anchor="middle" class="axl" transform="rotate(-90 16 {(T + H - B) / 2})">'
                 f'best objective so far (lower is better)</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Best objective so far against {escape(xlab)}" '
            f'class="trade" data-axis="{axis}"{"" if axis == "cost" else " hidden"}>{"".join(parts)}</svg>')


def render(runs: list[RunStats], title: str) -> str:
    q_best = min(runs, key=lambda r: r.best)
    cheap = min(runs, key=lambda r: r.cost)
    fast = min(runs, key=lambda r: r.seconds)
    headline = (f"{q_best.label} finds the best solver ({q_best.best:,.0f}, {q_best.gain_pct:.1f}% better than the seed); "
                f"{cheap.label} gets {cheap.gain_pct:.1f}% for {fmt(cheap.cost, 'usd')}"
                + (f" in {fmt(fast.seconds, 'min')}" if fast is cheap else "") + ".")
    quality = (bars("Gain over the seed (validate)", "higher is better", runs, lambda r: r.gain_pct, "pct", True)
               + bars("Gain over the seed (holdout, never searched)", "higher is better", runs, lambda r: r.holdout_gain_pct, "pct", True))
    efficiency = (bars("Total cost", "lower is better", runs, lambda r: r.cost, "usd", False)
                  + bars("Total wall clock", "lower is better", runs, lambda r: r.seconds, "min", False)
                  + bars("Objective points gained per dollar", "higher is better", runs, lambda r: r.gain_per_dollar, "num", True))
    rows = [
        ("Model", [r.model for r in runs], None),
        ("Best on validate (lower is better)", [fmt(r.best, "num") for r in runs], min(runs, key=lambda r: r.best)),
        ("Gain over the seed", [fmt(r.gain_pct, "pct") for r in runs], max(runs, key=lambda r: r.gain_pct)),
        (f"Gain over the best classical solver ({runs[0].classical[0]})",
         [fmt(100 * (r.classical[1] - r.best) / r.classical[1], "pct") for r in runs], min(runs, key=lambda r: r.best)),
        ("Best on holdout (lower is better)", [fmt(r.holdout_best, "num") for r in runs], min(runs, key=lambda r: r.holdout_best)),
        ("Total cost", [fmt(r.cost, "usd") for r in runs], cheap),
        ("Total wall clock", [fmt(r.seconds, "min") for r in runs], fast),
        ("Agent tokens", [f"{r.tokens / 1e6:.2f} M" for r in runs], min(runs, key=lambda r: r.tokens)),
        ("Objective points per dollar", [fmt(r.gain_per_dollar, "num") for r in runs], max(runs, key=lambda r: r.gain_per_dollar)),
    ]
    head = "".join(f'<th><span class="sw" style="background:var({r.color})"></span>{escape(r.label)}</th>' for r in runs)
    body = "".join(
        f"<tr><td class='l'>{escape(name)}</td>"
        + "".join(f"<td class='{'win' if win is r else ''}'>{escape(v)}</td>" for r, v in zip(runs, vals)) + "</tr>"
        for name, vals, win in rows)
    legend = "".join(f'<span><span class="sw" style="background:var({r.color})"></span>{escape(r.label)}</span>' for r in runs)
    r0 = runs[0]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title><style>
:root{{color-scheme:light;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);
--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}}}
:root[data-theme="dark"]{{color-scheme:dark;--surface:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70}}
*{{box-sizing:border-box}}[hidden]{{display:none!important}}body{{margin:0;background:var(--surface);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;padding:32px 20px 56px}}
main{{max-width:1080px;margin:0 auto;display:grid;gap:28px}}h1{{font-size:26px;line-height:1.2;margin:0;text-wrap:balance}}h2{{font-size:17px;margin:0 0 4px}}
.lead{{color:var(--ink2);margin:6px 0 0;max-width:75ch}}.groups{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}}
.group{{border:1px solid var(--ring);border-radius:10px;padding:16px;display:grid;gap:14px;align-content:start}}.group h2 span{{color:var(--ink2);font-weight:400}}
figure{{margin:0}}.mini figcaption{{display:flex;justify-content:space-between;gap:8px;font-size:13.5px}}.mini figcaption span{{color:var(--muted)}}
svg{{width:100%;height:auto;display:block}}text{{fill:var(--ink2);font-size:12px;font-variant-numeric:tabular-nums}}
.lab{{fill:var(--ink2);font-size:13px}}.win{{fill:var(--ink);font-weight:700}}.val{{font-size:12.5px}}.axis{{stroke:var(--axis);stroke-width:1}}.grid{{stroke:var(--grid);stroke-width:1}}
.ref{{stroke:var(--muted);stroke-width:1;stroke-dasharray:4 4}}.refl{{fill:var(--muted);font-size:11.5px}}.tick{{fill:var(--muted);font-size:11.5px}}.axl{{fill:var(--ink2);font-size:12px}}
.line{{fill:none;stroke-width:2}}.dot{{stroke:var(--surface);stroke-width:2}}.endl{{fill:var(--ink);font-size:12.5px;font-weight:600}}
.tabs{{display:flex;gap:6px;margin:8px 0}}.tabs button{{font:inherit;font-size:13px;border:1px solid var(--axis);background:none;color:var(--ink2);border-radius:999px;padding:3px 12px;cursor:pointer}}
.tabs button.on{{background:var(--ink);color:var(--surface);border-color:var(--ink)}}.legend{{display:flex;gap:16px;font-size:13px;color:var(--ink2)}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:baseline}}
.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}}th,td{{padding:7px 10px;border-bottom:1px solid var(--grid);text-align:right}}
th{{color:var(--ink2);font-weight:600}}td.l,th.l{{text-align:left;color:var(--ink2)}}td.win{{font-weight:700}}td.win::after{{content:" ★"}}
.note{{color:var(--muted);font-size:13px;max-width:90ch}}#tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12.5px;padding:5px 8px;border-radius:6px;max-width:320px}}
</style></head><body><main>
<header><h1>{escape(headline)}</h1><p class="lead">{escape(title)}. Same settings and seed for every model; objective = total edit distance on the
validate instances (four DNA instances of 280–688 bp and one 320 aa protein). Seed solver {r0.seed:,.0f}; planted reference {r0.planted:,.0f} (the string the instances were generated from, not a proven optimum).</p></header>
<section class="groups">
<div class="group"><h2>Quality <span>· {escape(q_best.label)} wins</span></h2>{quality}</div>
<div class="group"><h2>Efficiency <span>· {escape(cheap.label)} wins</span></h2>{efficiency}</div>
</section>
<section><h2>Best solver found, against money and time spent</h2>
<p class="lead">Each step is the end of a generation. A line that ends lower is a better solver; a line that ends further left was cheaper or faster.</p>
<div class="tabs"><button class="on" data-axis="cost">vs cost</button><button data-axis="time">vs wall clock</button></div>
<div class="legend">{legend}</div>
<div class="scroll">{tradeoff(runs, "cost")}{tradeoff(runs, "time")}</div></section>
<section><h2>All numbers</h2><div class="scroll"><table><thead><tr><th class="l"></th>{head}</tr></thead><tbody>{body}</tbody></table></div></section>
<p class="note">One run per model, so differences of a few percent are within noise. Cost is Claude Code's own estimate
(<code>total_cost_usd</code>), including cache reads and writes and output tokens. Holdout = instances generated from seeds the search
never used; seed solver on holdout {r0.holdout_seed:,.0f}.</p>
</main><div id="tip" hidden></div><script>
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{{const t=e.target.closest&&e.target.closest('[data-tip]');
if(!t){{tip.hidden=true;return}}tip.textContent=t.dataset.tip;tip.hidden=false;tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px'}});
document.querySelectorAll('.tabs button').forEach(b=>b.addEventListener('click',()=>{{
document.querySelectorAll('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
document.querySelectorAll('svg.trade').forEach(s=>s.hidden=s.dataset.axis!==b.dataset.axis)}}));
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", help="label=run_dir, at most 3")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--title", default="Model comparison on the autoresearch swarm")
    args = p.parse_args()
    if len(args.runs) > len(SERIES):
        p.error(f"at most {len(SERIES)} runs")
    runs = []
    for spec, color in zip(args.runs, SERIES):
        label, path = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        runs.append(load(label, Path(path), color))
    Path(args.out).write_text(render(runs, args.title))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
