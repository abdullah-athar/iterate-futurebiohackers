"""Model-grid page: many runs (configurations × benchmarks × repeats) on quality-vs-efficiency scatters.

A configuration is named from the models its agents used, in order: one model gives its name ("Sonnet"),
a hand-over gives "Sonnet → Opus", and Sonnet → Opus is called "Sopus". Runs of the same configuration on
the same benchmark are repeats: their mean and spread are what the page compares. A run made by a benchmark
script (a `bench.json` with an `arm`, e.g. A / B / Cfix) is named after its arm instead."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from html import escape

from itertools import combinations

from .charts import BandGroup, DotColumn, ScatterGroup, _mean_sd, band_chart, fmt_num, paired_dots, scatter_chart
from .html import CSS, JS, PALETTE
from .load import Run
from .metrics import Summary, held_out_gain_pct, summarize

CONFIG_COLORS = {"Haiku": PALETTE[0], "Sonnet": PALETTE[1], "Sopus": PALETTE[3], "Opus": PALETTE[2]}  # page order: cheapest first
NICKNAMES = {"Sonnet → Opus": "Sopus"}

GRID_CSS = """
.facets{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),540px));gap:16px}
.facet{border:1px solid var(--line);border-radius:10px;padding:10px 12px 4px}.facet h3{margin:2px 0 4px}
td.win{font-weight:700}
table.h2h{margin:2px 0 10px;font-size:12.5px}table.h2h td,table.h2h th{padding:4px 6px}table.h2h td:first-child{min-width:0}
"""

GRID_JS = """
function paretoVisibility(){document.querySelectorAll('.pareto').forEach(p=>p.style.display=HIDDEN.size?'none':'');}
document.querySelectorAll('.runtoggles button').forEach(b=>b.addEventListener('click',paretoVisibility));
"""


def bench_info(run: Run) -> dict:
    try:
        return json.loads((run.path / "bench.json").read_text())
    except (OSError, ValueError):
        return {}


def config_name(run: Run) -> str:
    if arm := bench_info(run).get("arm"):
        return str(arm)
    models = []
    for e in run.proposals:
        m = e.proposer.split(":", 1)[-1]
        m = m[:1].upper() + m[1:]
        if not models or models[-1] != m:
            models.append(m)
    name = " → ".join(models) or "?"
    return NICKNAMES.get(name, name)


def repeat_tag(run: Run) -> str:
    m = re.search(r"-s(\d+)$", run.path.name)
    return f"seed {m.group(1)}" if m else run.path.name


def total_cost(s: Summary, run: Run) -> float:
    """Agent cost plus the side LLM calls a run logged in `llm_usage.jsonl` (e.g. solver descriptions)."""
    extra = 0.0
    try:
        for line in (run.path / "llm_usage.jsonl").read_text().splitlines():
            if line.strip():
                extra += float(json.loads(line).get("cost_usd") or 0)
    except (OSError, ValueError):
        pass
    return s.cost_usd + extra or math.nan


def points_per_dollar(s: Summary, run: Run) -> float:
    cost = total_cost(s, run)
    return (s.seed_objective - s.best_objective) / cost if math.isfinite(cost) else math.nan


def gain_by_generation(run: Run, s: Summary) -> list[float]:
    """Best gain over the seed after each generation (index 0 = the seed solver itself)."""
    gens = max((e.generation or 0 for e in run.proposals), default=0)
    out = []
    for g in range(gens + 1):
        objs = [e.objective for e in run.entries if e.scored and e.confirmed is not False and (e.is_seed or (e.generation or 0) <= g)]
        out.append(100.0 * (s.seed_objective - min(objs)) / s.seed_objective if objs and s.seed_objective else math.nan)
    return out


def switch_generation(run: Run) -> tuple[int, str] | None:
    """(first generation on the new model, "Sonnet → Opus") when the run's proposer model changes."""
    first = None
    for e in run.proposals:
        m = e.proposer.split(":", 1)[-1]
        m = m[:1].upper() + m[1:]
        if first is None:
            first = m
        elif m != first and e.generation:
            return e.generation, f"{first} → {m}"
    return None


def _tabbed(group: str, tabs: list[tuple[str, str, str]]) -> str:
    buttons = "".join(f'<button class="{"on" if i == 0 else ""}" data-key="{k}">{escape(lab)}</button>' for i, (k, lab, _) in enumerate(tabs))
    panels = "".join(f'<div class="panel {"on" if i == 0 else ""}" data-group="{group}" data-key="{k}">{body}</div>'
                     for i, (k, _, body) in enumerate(tabs))
    return f'<div class="tabs" data-group="{group}">{buttons}</div>{panels}'


MEASURES = [  # (key, column title, unit, higher is better, value)
    ("gain", "gain over the seed", "pct", True, lambda s, r: s.gain_vs_seed_pct),
    ("held", "held-out gain", "pct", True, held_out_gain_pct),
    ("cost", "cost", "usd", False, total_cost),
    ("time", "wall-clock", "seconds", False, lambda s, r: s.seconds),
    ("ppd", "points per dollar", "", True, points_per_dollar),
]
VIEWS = [  # (tab key, tab label, y measure, x measure)
    ("gain-cost", "gain vs cost", "gain", "cost"),
    ("gain-time", "gain vs wall-clock", "gain", "time"),
    ("held-cost", "held-out gain vs cost", "held", "cost"),
]


def render_grid(runs: list[Run], title: str = "Autoresearch — model grid across benchmarks") -> str:
    done = [r for r in runs if any(e.get("type") == "run_end" for e in r.events) or not r.events]
    rows = []  # (benchmark, config, run, summary, values)
    for r in done:
        s = summarize(r)
        if math.isfinite(s.best_objective):
            rows.append((r.problem, config_name(r), r, s, {k: f(s, r) for k, _, _, _, f in MEASURES}))
    benches = sorted({b for b, *_ in rows})
    configs = sorted({c for _, c, *_ in rows}, key=lambda c: (list(CONFIG_COLORS).index(c) if c in CONFIG_COLORS else 99, c))
    extra = iter(PALETTE[4:] + PALETTE)
    colors = {c: CONFIG_COLORS.get(c) or next(extra) for c in configs}
    units = {k: u for k, _, u, _, _ in MEASURES}
    titles = {k: t for k, t, _, _, _ in MEASURES}

    toggles = ("<div class='runtoggles' role='group' aria-label='Configurations shown'><span>Show:</span>"
               + "".join(f"<button class='on' aria-pressed='true' data-run='{escape(c, quote=True)}'>"
                         f"<span class='sw' style='background:{colors[c]}'></span>{escape(c)}</button>" for c in configs)
               + "</div>") if len(configs) > 1 else ""

    panels, buttons = [], []
    for i, (key, label, ym, xm) in enumerate(VIEWS):
        facets = []
        for b in benches:
            groups = []
            for c in configs:
                pts = [(v[xm], v[ym], f"{c} · {repeat_tag(r)}: {fmt_num(v[ym], 'pct')} {titles[ym]}, "
                        f"{fmt_num(v[xm], units[xm])}, best {fmt_num(s.best_objective)}")
                       for bb, cc, r, s, v in rows if bb == b and cc == c]
                if pts:
                    groups.append(ScatterGroup(c, colors[c], pts))
            n = sum(len(g.points) for g in groups)
            facets.append(f"<div class='facet'><h3>{escape(b)} · {n} run{'s' if n != 1 else ''}</h3>"
                          + scatter_chart(groups, x_label=f"{titles[xm]} (lower is better)", y_label=f"{titles[ym]} (higher is better)",
                                          x_unit=units[xm], y_unit="pct")
                          + "</div>")
        panels.append(f'<div class="panel {"on" if i == 0 else ""}" data-group="grid" data-key="{key}"><div class="facets">{"".join(facets)}</div></div>')
        buttons.append(f'<button class="{"on" if i == 0 else ""}" data-key="{key}">{escape(label)}</button>')
    scatter_card = (
        "<section class='card'><h2>Quality vs efficiency across benchmarks</h2><p class='lead'>One panel per benchmark. "
        "Each faint dot is one run, the large dot is the mean of a configuration's runs with ±1 standard deviation whiskers. "
        "Up and to the left is better; the dashed line joins the configurations no other one beats on both axes "
        "(hidden while a configuration is toggled off). Each panel has its own scales.</p>"
        f'<div class="tabs" data-group="grid">{"".join(buttons)}</div>' + "".join(panels) + "</section>"
    )

    # Per-seed view: every repeat visible, the same seed joined across configurations, plus paired head-to-heads.
    seed_tabs = []
    for key, t, u, higher, _ in MEASURES:
        facets = []
        for b in benches:
            cols = []
            for c in configs:
                pts = {repeat_tag(r): (v[key], f"{c} · {repeat_tag(r)}: {fmt_num(v[key], u)} {t}") for bb, cc, r, _, v in rows if bb == b and cc == c}
                if pts:
                    cols.append(DotColumn(c, colors[c], pts))
            h2h = []
            for a, z in (combinations(cols, 2) if len(cols) <= 3 else ((cols[0], z) for z in cols[1:])):
                common = sorted(tag for tag in set(a.points) & set(z.points)
                                if math.isfinite(a.points[tag][0]) and math.isfinite(z.points[tag][0]))
                if not common:
                    continue
                diffs = [z.points[tag][0] - a.points[tag][0] for tag in common]
                wins = sum(1 for d in diffs if (d > 0 if higher else d < 0))
                md = sum(diffs) / len(diffs)
                h2h.append(f"<tr><td class='l'><span class='sw' style='background:{z.color}'></span>{escape(z.label)} vs "
                           f"<span class='sw' style='background:{a.color}'></span>{escape(a.label)}</td>"
                           f"<td>{z.label} better on {wins}/{len(common)} seeds</td>"
                           f"<td>{f'{md:+.1f} pts' if u == 'pct' else ('+' if md >= 0 else '−') + fmt_num(abs(md), u)} "
                           "<span class='muted'>mean paired difference</span></td></tr>")
            n = sum(len(c.points) for c in cols)
            facets.append(f"<div class='facet'><h3>{escape(b)} · {n} run{'s' if n != 1 else ''}</h3>"
                          + paired_dots(cols, unit=u, higher_is_better=higher, from_zero=key in ("cost", "time", "ppd"))
                          + (f"<table class='h2h'>{''.join(h2h)}</table>" if h2h else "") + "</div>")
        seed_tabs.append((key, t, f"<div class='facets'>{''.join(facets)}</div>"))
    seed_card = ("<section class='card'><h2>Every seed, side by side</h2><p class='lead'>One dot per run, labelled with its swarm seed; the "
                 "coloured bar is the mean. A grey line joins the same seed across configurations, so parallel lines mean the "
                 "configurations rank the same way on every seed. Under each panel, the head-to-head counts the seeds where one "
                 "configuration beat the other with the same seed, and the mean of those per-seed differences (every pair for up to three "
                 "configurations, otherwise each one against the first).</p>"
                 + _tabbed("seeds", seed_tabs) + "</section>")

    # Progress by generation: mean line, min–max band over the seeds, each seed as a faint line.
    facets = []
    for b in benches:
        groups, switches = [], {}
        for c in configs:
            sel = [(r, s) for bb, cc, r, s, _ in rows if bb == b and cc == c]
            if sel:
                groups.append(BandGroup(c, colors[c], {repeat_tag(r): gain_by_generation(r, s)[1:] for r, s in sel}))
                sw = switch_generation(sel[0][0])
                if sw:
                    switches[(sw[0] - 0.5, f"{c}: {sw[1]}")] = colors[c]
        n_gen = max((len(v) for g in groups for v in g.runs.values()), default=0)
        for g in groups:  # a run that stopped early (e.g. its spend cap) keeps its final best
            g.runs = {t: v + v[-1:] * (n_gen - len(v)) for t, v in g.runs.items()}
        facets.append(f"<div class='facet'><h3>{escape(b)}</h3>"
                      + band_chart(groups, x_label="generation", y_label="best gain over the seed so far (higher is better)", y_unit="pct", x0=1,
                                   switches=[(x, lab, col) for (x, lab), col in switches.items()]) + "</div>")
    progress_card = ("<section class='card'><h2>Progress by generation</h2><p class='lead'>Best gain over the seed solver after each "
                     "generation (validate split), from generation 1 so the gaps stay readable (every run starts from +0%). The thick line is the mean over the seeds, the shaded band spans the best and "
                     "worst seed, and each faint line is one seed. A dashed line marks a model hand-over. A run that stopped "
                     "earlier than the others (e.g. on its spend cap) keeps its final value.</p>"
                     f"<div class='facets'>{''.join(facets)}</div></section>")

    arms = {}
    for _, c, r, _, _ in rows:
        info = bench_info(r)
        if info.get("flags") is not None:
            arms.setdefault(c, (info.get("flags") or "(defaults)", info.get("sha", ""), info.get("model", "")))
    arms_card = ("<section class='card'><h2>Configurations</h2><p class='lead'>Flags each arm was run with (from its "
                 "<code>bench.json</code>).</p><table><tr><th class='l'>arm</th><th class='l'>flags</th><th class='l'>model</th>"
                 "<th class='l'>commit</th></tr>" + "".join(
                     f"<tr data-run='{escape(c, quote=True)}'><td class='l'><span class='sw' style='background:{colors[c]}'></span><b>{escape(c)}</b></td>"
                     f"<td class='l' style='white-space:normal'><code>{escape(f)}</code></td><td class='l'>{escape(m)}</td><td class='l'><code>{escape(sha)}</code></td></tr>"
                     for c, (f, sha, m) in sorted(arms.items(), key=lambda kv: configs.index(kv[0])))
                 + "</table></section>") if arms else ""

    head = "<tr><th class='l'>benchmark</th><th class='l'>configuration</th><th>runs</th><th>best objective</th>" + "".join(
        f"<th>{escape(t)}</th>" for _, t, _, _, _ in MEASURES) + "</tr>"
    body = []
    for b in benches:
        stats = {}
        for c in configs:
            sel = [(s, v) for bb, cc, _, s, v in rows if bb == b and cc == c]
            if sel:
                stats[c] = (len(sel), _mean_sd([s.best_objective for s, _ in sel]),
                            {k: _mean_sd([v[k] for _, v in sel]) for k, *_ in MEASURES})
        best = {}
        for k, _, _, higher, _ in MEASURES:
            means = [(m[2][k][0], c) for c, m in stats.items() if math.isfinite(m[2][k][0])]
            if means:
                best[k] = (max if higher else min)(means)[1]
        for c, (n, (bo, bo_sd), ms) in stats.items():
            cells = []
            for k, _, u, _, _ in MEASURES:
                m, sd = ms[k]
                txt = fmt_num(m, u) + (f" <span class='muted'>± {fmt_num(sd, u).lstrip('+')}</span>" if n > 1 and math.isfinite(sd) else "")
                win = best.get(k) == c and len(stats) > 1
                cells.append(f"<td class='{'win' if win else ''}'>{txt}{' ★' if win else ''}</td>")
            body.append(f"<tr data-run='{escape(c, quote=True)}'><td class='l'>{escape(b)}</td><td class='l'><span class='sw' "
                        f"style='background:{colors[c]}'></span><b>{escape(c)}</b></td><td>{n}</td>"
                        f"<td>{fmt_num(bo)}{f' <span class=muted>± {fmt_num(round(bo_sd))}</span>' if n > 1 else ''}</td>{''.join(cells)}</tr>")
    table_card = ("<section class='card'><h2>Mean ± standard deviation per benchmark</h2><p class='lead'>★ marks the best mean of each "
                  "column within a benchmark (over all configurations, whatever is toggled). Gains are relative to the seed solver; "
                  "the held-out gain is measured on instances the search never saw.</p>"
                  f"<div style='overflow-x:auto'><table>{head}{''.join(body)}</table></div></section>")

    pending = len(runs) - len(done)
    sub = (f"{len(rows)} finished run{'s' if len(rows) != 1 else ''} · {len(benches)} benchmark(s) · {len(configs)} configuration(s)"
           + (f" · {pending} still running" if pending else "") + f" · updated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    return "\n".join([
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>Model grid</title><style>{CSS}{GRID_CSS}</style></head><body><main>",
        f"<h1>{escape(title)}</h1><p class='sub'>{sub}</p>",
        toggles, arms_card, seed_card, progress_card, scatter_card, table_card,
        "<footer>Rendered by <code>python -m autoresearch_viz grid</code> from "
        + ", ".join(f"<code>{escape(r.path.name)}</code>" for r in done) + "</footer>",
        f"</main><div id='tip'></div><script>{JS}{GRID_JS}</script></body></html>",
    ])
