"""Model-grid page: many runs (configurations × benchmarks × repeats) on quality-vs-efficiency scatters.

A configuration is named from the models its agents used, in order: one model gives its name ("Sonnet"),
a hand-over gives "Sonnet → Opus", and Sonnet → Opus is called "Sopus". Runs of the same configuration on
the same benchmark are repeats: their mean and spread are what the page compares."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from html import escape

from .charts import ScatterGroup, _mean_sd, fmt_num, scatter_chart
from .html import CSS, JS, PALETTE
from .load import Run
from .metrics import Summary, held_out_gain_pct, summarize

CONFIG_COLORS = {"Haiku": PALETTE[0], "Sonnet": PALETTE[1], "Opus": PALETTE[2], "Sopus": PALETTE[3]}
NICKNAMES = {"Sonnet → Opus": "Sopus"}

GRID_CSS = """
.facets{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),540px));gap:16px}
.facet{border:1px solid var(--line);border-radius:10px;padding:10px 12px 4px}.facet h3{margin:2px 0 4px}
td.win{font-weight:700}
"""

GRID_JS = """
function paretoVisibility(){document.querySelectorAll('.pareto').forEach(p=>p.style.display=HIDDEN.size?'none':'');}
document.querySelectorAll('.runtoggles button').forEach(b=>b.addEventListener('click',paretoVisibility));
"""


def config_name(run: Run) -> str:
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


def points_per_dollar(s: Summary) -> float:
    return (s.seed_objective - s.best_objective) / s.cost_usd if s.cost_usd else math.nan


MEASURES = [  # (key, column title, unit, higher is better, value)
    ("gain", "gain over the seed", "pct", True, lambda s, r: s.gain_vs_seed_pct),
    ("held", "held-out gain", "pct", True, held_out_gain_pct),
    ("cost", "agent cost", "usd", False, lambda s, r: s.cost_usd or math.nan),
    ("time", "wall-clock", "seconds", False, lambda s, r: s.seconds),
    ("ppd", "points per dollar", "", True, lambda s, r: points_per_dollar(s)),
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
        toggles, scatter_card, table_card,
        "<footer>Rendered by <code>python -m autoresearch_viz grid</code> from "
        + ", ".join(f"<code>{escape(r.path.name)}</code>" for r in done) + "</footer>",
        f"</main><div id='tip'></div><script>{JS}{GRID_JS}</script></body></html>",
    ])
