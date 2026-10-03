"""Assemble the self-contained dashboard HTML (inline CSS + JS, no external assets)."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from html import escape

from .charts import BarGroup, RefLine, Series, fmt_num, grouped_hbars, stacked_hbars, step_chart
from .load import STATUS_LABELS, STATUS_ORDER, STATUS_REJECTED_DUPLICATE, Run
from .metrics import Summary, best_delta_text, best_so_far, per_instance, summarize

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]
STATUS_COLORS = {
    "kept": "#16a34a",
    "evaluated": "#94a3b8",
    "rejected_screen": "#f59e0b",
    "rejected_duplicate": "#fbbf24",
    "failed": "#ef4444",
    "rejected_guard": "#7c2d12",
    "seed": "#2563eb",
}
MODE_COLORS = {"tune": "#0ea5e9", "fix_losers": "#f97316", "new_family": "#8b5cf6", "merge": "#ec4899", "seed": "#2563eb"}
VERDICT_COLORS = {"supported": "#16a34a", "partial": "#65a30d", "falsified": "#94a3b8", "unconfirmed": "#dc2626",
                  "inconclusive": "#f59e0b", "untested": "#cbd5e1"}

CSS = """
:root{--bg:#f8fafc;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--accent:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:1280px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:19px;margin:0 0 4px}h3{font-size:15px;margin:14px 0 6px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.sub{color:var(--muted);margin:0 0 22px}.muted{color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:0 0 20px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.card p.lead{margin:0 0 12px;color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:20px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:28px;font-weight:700;letter-spacing:-.02em}.kpi .k{color:var(--muted);font-size:13px}.kpi .d{font-size:12px;color:var(--muted);margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:6px 0 10px;font-size:13px}
.legend span.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
.tabs{display:flex;gap:6px;margin:0 0 8px}.tabs button{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 12px;font:inherit;font-size:13px;cursor:pointer;color:var(--muted)}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.panel{display:none}.panel.on{display:block}
svg.chart{max-width:100%;height:auto;display:block}
svg .grid{stroke:#eef2f7;stroke-width:1}svg .axis{stroke:#cbd5e1;stroke-width:1}
svg .tick,svg .val,svg .seg{font-size:11px;fill:#64748b}svg .seg{fill:#fff;font-weight:600}svg .label{font-size:12px;fill:#334155}
svg .ref{font-size:11px}svg .cat{font-size:12px;fill:#0f172a}
svg [data-tip]{cursor:pointer}svg circle[data-tip]:hover{r:7}
table{width:100%;border-collapse:collapse;font-size:13.5px}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top}
th:first-child,td:first-child{text-align:left}td{white-space:nowrap}td:first-child{white-space:normal;min-width:240px}td:first-child .note{white-space:normal}th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.l,th.l{text-align:left}tr.best td{background:#f0fdf4}td .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:11.5px;font-weight:600;color:#fff;white-space:nowrap}
.traj td.h{text-align:left;max-width:560px;white-space:normal}.traj tr.kept td{background:#f0fdf4}.traj tr.dup td{color:#94a3b8}.traj tr.fail td.h{color:#b91c1c}
.traj .note{display:block;font-size:12px;color:var(--muted)}
select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:#fff}
#tip{position:fixed;pointer-events:none;background:#0f172a;color:#fff;padding:6px 9px;border-radius:6px;font-size:12px;max-width:360px;display:none;z-index:9;white-space:pre-line}
.note-syn{background:#fffbeb;border:1px solid #fde68a;color:#92400e;border-radius:10px;padding:10px 14px;margin-bottom:18px;font-size:13.5px}
footer{color:var(--muted);font-size:12px;margin-top:30px}
.events{margin:0;padding-left:18px;font-size:12.5px;color:#334155}.events li{margin:2px 0}
@media print{.tabs{display:none}.panel{display:block!important}}
"""

JS = """
function bind(){
document.querySelectorAll('.tabs').forEach(t=>{t.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
 t.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 const grp=t.dataset.group;document.querySelectorAll('.panel[data-group="'+grp+'"]').forEach(p=>p.classList.toggle('on',p.dataset.key===b.dataset.key));})); });
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
 tip.style.left=Math.min(e.clientX+14,window.innerWidth-380)+'px';tip.style.top=(e.clientY+14)+'px';});el.addEventListener('mouseleave',()=>tip.style.display='none');});
const sel=document.getElementById('runsel');if(sel){sel.addEventListener('change',()=>{document.querySelectorAll('.traj').forEach(p=>p.classList.toggle('on',p.dataset.key===sel.value));});}
}
bind();
"""


def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{escape(text)}</span>'


def _legend(items: list[tuple[str, str]]) -> str:
    return (
        '<div class="legend">'
        + "".join(f'<span><span class="sw" style="background:{c}"></span>{escape(l)}</span>' for l, c in items)
        + "</div>"
    )


def _kpis(summaries: list[Summary], runs: list[Run]) -> str:
    scored = [s for s in summaries if math.isfinite(s.gain_vs_baseline_pct)]
    if not scored:
        return ""
    best = max(scored, key=lambda s: s.gain_vs_baseline_pct)
    dupes = sum(s.counts.get(STATUS_REJECTED_DUPLICATE, 0) for s in summaries)
    tiles = [
        (fmt_num(best.gain_vs_baseline_pct, "pct"), "best improvement vs set-median baseline", f"{best.label} · {best.objective_split} split"),
    ]
    # the headline that matters: how far the agents got past the strongest classical solver, same CPU budget
    run = next((r for r in runs if r.label == best.label), None)
    classical = [(name, ev[run.objective_split].score) for name, ev in (run.baselines if run else {}).items()
                 if run.objective_split in ev and ev[run.objective_split].ok]
    if classical and math.isfinite(best.best_objective):
        name, score = min(classical, key=lambda c: c[1])
        pct = 100.0 * (score - best.best_objective) / max(score, 1.0)
        tiles.append((fmt_num(pct, "pct"), f"vs strongest classical solver ({name})",
                      f"{fmt_num(score)} → agents' best {fmt_num(best.best_objective)} · same CPU budget"))
    tiles.append((fmt_num(best.tokens, "tokens"), "agent tokens used", f"{best.evaluated} evaluations in {fmt_num(best.seconds, 'seconds')}"
                  + (f" · ${best.cost_usd:.2f}" if best.cost_usd else "")))
    if best.tokens_per_pct:
        tiles.append((fmt_num(best.tokens_per_pct, "tokens"), "tokens per 1% gained", f"{best.label} · research efficiency"))
    if best.holdout_pct is not None:
        tiles.append((fmt_num(best.holdout_pct, "pct"), "held-out improvement (unseen instances)", best.label))
    tiles.append((str(dupes), "duplicate proposals skipped", "evaluations saved by the novelty gate; their proposal tokens were still spent"))
    return (
        '<div class="kpis">'
        + "".join(
            f'<div class="kpi"><div class="v">{escape(v)}</div><div class="k">{escape(k)}</div><div class="d">{escape(d)}</div></div>'
            for v, k, d in tiles
        )
        + "</div>"
    )


def _progress_card(runs: list[Run], colors: dict[str, str], summaries: list[Summary]) -> str:
    refs: list[RefLine] = []
    s0 = next((s for s in summaries if math.isfinite(s.baseline)), None)
    if s0:
        refs.append(RefLine(s0.baseline, "set-median baseline", "#64748b"))
        if s0.best_known is not None:
            refs.append(RefLine(s0.best_known, "planted reference (not proven optimal)", "#16a34a", "2 4"))
    classical = [(name, ev[r.objective_split].score) for r in runs[:1] for name, ev in r.baselines.items()
                 if name != "set_median" and r.objective_split in ev and ev[r.objective_split].ok]
    if classical:
        name, score = min(classical, key=lambda c: c[1])
        refs.append(RefLine(score, f"best classical solver ({name})", "#b45309", "4 3"))
    axes = [
        ("evals", "evaluations (duplicates skipped by the novelty gate don't count)", ""),
        ("tokens", "cumulative agent tokens (prompt + completion)", "tokens"),
        ("seconds", "wall-clock", "seconds"),
    ]
    panels, buttons = [], []
    for i, (key, xl, unit) in enumerate(axes):
        series = []
        for run in runs:
            pts = best_so_far(run)
            xy = [(getattr(p, key), p.best) for p in pts]
            marks = []
            for p in pts:
                if p.improved:
                    e = next(x for x in run.entries if x.id == p.entry_id)
                    marks.append((getattr(p, key), p.best, f"#{e.id} [{e.mode}] → {fmt_num(p.best)}\n{e.hypothesis}"))
            series.append(Series(run.label, colors[run.label], xy, marks))
        max_x = max((getattr(p, key) for r in runs for p in best_so_far(r)), default=0)
        panels.append(
            f'<div class="panel {"on" if i == 0 else ""}" data-group="x" data-key="{key}">'
            + step_chart(
                series, refs=refs, x_label=xl, y_label="objective (total edit distance, lower is better)", x_unit=unit, extend_to=max_x
            )
            + "</div>"
        )
        buttons.append(f'<button class="{"on" if i == 0 else ""}" data-key="{key}">vs {key if key != "seconds" else "wall-clock"}</button>')
    return (
        '<section class="card"><h2>Research progress</h2><p class="lead">Best objective found so far. Dots mark proposals that set a new global best; hover for the hypothesis.</p>'
        f'<div class="tabs" data-group="x">{"".join(buttons)}</div>'
        + _legend([(r.label, colors[r.label]) for r in runs])
        + "".join(panels)
        + "</section>"
    )


def _baselines_card(runs: list[Run]) -> str:
    rows = []
    for run in runs:
        best = run.best
        for name, ev in sorted(run.baselines.items(), key=lambda kv: kv[1][run.objective_split].score):
            v, h = ev.get(run.objective_split), ev.get("holdout")
            gap = (f"{100 * (v.score - best.objective) / max(v.score, 1):+.1f}%" if best and v and v.ok else "—")
            rows.append(f"<tr><td class='l'>{escape(run.label)}</td><td class='l'><b>{escape(name)}</b></td>"
                        f"<td>{fmt_num(v.score) if v and v.ok else 'fail'}</td><td>{gap}</td>"
                        f"<td>{fmt_num(h.score) if h and h.ok else '—'}</td></tr>")
        if best:
            hold = best.evals.get("holdout")
            rows.append(f"<tr class='best'><td class='l'>{escape(run.label)}</td><td class='l'><b>agents' best (#{best.id})</b></td>"
                        f"<td>{fmt_num(best.objective)}</td><td>—</td><td>{fmt_num(hold.score) if hold and hold.ok else '—'}</td></tr>")
    if not rows:
        return ""
    return ("<section class='card'><h2>Classical baselines</h2><p class='lead'>Non-agent solvers from "
            "<code>median_string/solvers</code>, scored once per run under the same CPU budget. "
            "Δ = how much worse each is than the agents' best.</p><table><tr><th class='l'>run</th><th class='l'>solver</th>"
            "<th>objective</th><th>Δ vs agents' best</th><th>holdout</th></tr>" + "".join(rows) + "</table></section>")


def _scoreboard(summaries: list[Summary], colors: dict[str, str], runs: list[Run]) -> str:
    best_label = max((s for s in summaries if math.isfinite(s.gain_vs_baseline_pct)), key=lambda s: s.gain_vs_baseline_pct, default=None)
    head = (
        "<tr><th class='l'>flavour</th><th>best objective (seed)</th><th>Δ vs baseline</th><th>gap closed</th><th>held-out Δ</th>"
        "<th>proposals</th><th>evaluated</th><th>kept</th><th>dupes skipped</th><th>crashed</th><th>tokens</th><th>tokens / 1%</th><th>evals to 1st gain</th><th>wall-clock</th></tr>"
    )
    rows = []
    for s, run in zip(summaries, runs):
        gc = s.gap_closed_pct
        rows.append(
            f"<tr class='{'best' if s is best_label else ''}'><td class='l'><span class='sw' style='background:{colors[s.label]}'></span><b>{escape(s.label)}</b>"
            f"<span class='note' style='display:block;font-size:12px;color:#64748b'>{escape(run.describe())}</span></td>"
            f"<td>{fmt_num(s.best_objective)} <span class='muted'>({fmt_num(s.seed_objective)})</span></td>"
            f"<td><b>{fmt_num(s.gain_vs_baseline_pct, 'pct')}</b></td><td>{'—' if gc is None else f'{gc:.0f}%'}</td>"
            f"<td>{'—' if s.holdout_pct is None else fmt_num(s.holdout_pct, 'pct')}</td>"
            f"<td>{s.proposals}</td><td>{s.evaluated}</td><td>{s.counts.get('kept', 0)}</td><td>{s.counts.get('rejected_duplicate', 0)}</td><td>{s.counts.get('failed', 0)}</td>"
            f"<td>{fmt_num(s.tokens, 'tokens')}</td><td>{fmt_num(s.tokens_per_pct, 'tokens') if s.tokens_per_pct else '—'}</td>"
            f"<td>{'—' if s.evals_to_first_gain is None else s.evals_to_first_gain}</td><td>{fmt_num(s.seconds, 'seconds')}</td></tr>"
        )
    # prompt-mode hit rates
    modes = sorted({m for s in summaries for m in s.modes})
    mode_rows = []
    for s in summaries:
        cells = []
        for m in modes:
            st = s.modes.get(m)
            cells.append("<td>—</td>" if not st else f"<td>{st.global_wins}+{st.instance_wins} / {st.tried}</td>")
        mode_rows.append(
            f"<tr><td class='l'><span class='sw' style='background:{colors[s.label]}'></span>{escape(s.label)}</td>{''.join(cells)}</tr>"
        )
    mode_table = (
        (
            "<h3>Prompt-mode hit rate (global wins + per-instance wins / tried)</h3><table><tr><th class='l'>flavour</th>"
            + "".join(f"<th>{_badge(m, MODE_COLORS.get(m, '#64748b'))}</th>" for m in modes)
            + "</tr>"
            + "".join(mode_rows)
            + "</table>"
        )
        if modes
        else ""
    )
    return (
        "<section class='card'><h2>Flavour scoreboard</h2><p class='lead'>Lower objective is better. Δ is relative to the set-median baseline on the "
        f"{escape(summaries[0].objective_split)} split; held-out Δ is on fresh instances the loop never saw. 'tokens / 1%' = total agent tokens per percentage point gained over the seed.</p>"
        f"<div style='overflow-x:auto'><table>{head}{''.join(rows)}</table></div>{mode_table}</section>"
    )


def _instances_card(runs: list[Run], colors: dict[str, str]) -> str:
    rows = per_instance(runs)
    if not rows:
        return ""
    cats = [r.name for r in rows]
    groups = [
        BarGroup(
            run.label,
            colors[run.label],
            {r.name: r.best_by_run[run.label][0] for r in rows if run.label in r.best_by_run},
            {
                r.name: f"{run.label}: {fmt_num(r.best_by_run[run.label][0])} (entry #{r.best_by_run[run.label][1]}), baseline {fmt_num(r.baseline)}"
                + (f", planted reference {fmt_num(r.best_known)}" if r.best_known is not None else "")
                for r in rows
                if run.label in r.best_by_run
            },
        )
        for run in runs
    ]
    markers = {
        r.name: [(r.baseline, "set-median baseline", "#64748b")]
        + ([(r.best_known, "planted reference", "#16a34a")] if r.best_known is not None else [])
        for r in rows
    }
    legend = _legend(
        [(r.label, colors[r.label]) for r in runs] + [("baseline (dashed grey)", "#64748b"), ("planted reference (dashed green; a feasible answer, not a proven optimum)", "#16a34a")]
    )
    return (
        "<section class='card'><h2>Per-instance results</h2><p class='lead'>Best score each flavour reached on every benchmark instance. Shorter bars are better; the Pareto archive keeps solvers that win on any single instance, not just the total.</p>"
        + legend
        + grouped_hbars(cats, groups, markers=markers, x_label="edit distance (lower is better)")
        + "</section>"
    )


def _outcomes_card(summaries: list[Summary], colors: dict[str, str]) -> str:
    rows = [(s.label, [(STATUS_LABELS[st], STATUS_COLORS[st], float(s.counts.get(st, 0))) for st in STATUS_ORDER]) for s in summaries]
    legend = _legend([(STATUS_LABELS[st], STATUS_COLORS[st]) for st in STATUS_ORDER])
    return (
        "<section class='card'><h2>What happened to each proposal</h2><p class='lead'>Duplicates caught by the novelty gate were never evaluated, so they cost only the proposal tokens. Screen rejections failed the cheap small-tier cascade and never ran the expensive split.</p>"
        + legend
        + stacked_hbars(rows)
        + "</section>"
    )


def _trajectory_card(runs: list[Run], colors: dict[str, str]) -> str:
    options = "".join(f'<option value="{escape(r.label, quote=True)}">{escape(r.label)}</option>' for r in runs)
    panels = []
    for i, run in enumerate(runs):
        best = math.inf
        trs = []
        for e in run.entries:
            ev = run.objective_eval(e)
            delta = best_delta_text(run, e, best)
            cls = (
                "kept"
                if (e.improved_global or e.status == "kept")
                else ("dup" if e.status == STATUS_REJECTED_DUPLICATE else ("fail" if e.status == "failed" else ""))
            )
            detail = ""
            if ev and ev.instances:
                detail = "  ·  ".join(
                    f"{i.name.replace('dna_', '').replace('protein_', 'prot_')}: {fmt_num(i.score)}"
                    + ("↓" if i.name in e.improved_instances else "")
                    for i in ev.instances
                )
            elif e.status == STATUS_REJECTED_DUPLICATE:
                dup = e.novelty.get("duplicate_of")
                sim = e.novelty.get("similarity")
                detail = "near-duplicate" + (f" of #{dup}" if dup is not None else "") + (f" (similarity {sim})" if sim is not None else "")
            else:
                err = next((x.error for x in e.evals.values() if x.error), "")
                detail = err or e.note
            parents = ",".join(f"#{p}" for p in e.parent_ids) or "—"
            trs.append(
                f"<tr class='{cls}'><td>{e.id}</td><td class='muted'>{parents}</td><td>{_badge(e.mode or '?', MODE_COLORS.get(e.mode, '#64748b'))}</td>"
                f"<td>{_badge(STATUS_LABELS.get(e.status, e.status), STATUS_COLORS.get(e.status, '#64748b'))}"
                + (f"<br>{_badge(e.verdict, VERDICT_COLORS.get(e.verdict, '#64748b'))}" if e.verdict else "")
                + (f"<br><span class='muted'>gen {e.generation} · {e.usage.get('outcome', '')} · ${e.cost:.2f}</span>" if e.usage else "")
                + "</td>"
                f"<td>{fmt_num(e.objective) if e.scored else '—'}</td><td>{delta}</td><td>{fmt_num(e.tokens, 'tokens') if e.tokens else '—'}</td>"
                f"<td class='h'>{escape(e.hypothesis)}<span class='note'>{escape(detail)}</span></td></tr>"
            )
            if e.scored and e.objective < best:
                best = e.objective
        panels.append(
            f'<div class="traj panel {"on" if i == 0 else ""}" data-key="{escape(run.label, quote=True)}"><div style="overflow-x:auto"><table>'
            "<tr><th>#</th><th>parents</th><th>mode</th><th>outcome / verdict</th><th>objective</th><th>Δ best</th><th>tokens</th><th class='l'>hypothesis → what we learned</th></tr>"
            + "".join(reversed(trs[-TRAJECTORY_ROWS:]))
            + "</table></div></div>"
        )
    return (
        "<section class='card'><h2>Research trajectory</h2><p class='lead'>The ledger as a lab notebook, newest first (last "
        f"{TRAJECTORY_ROWS}): each proposal's hypothesis, the prompt mode that produced it, and what the evaluation taught us. "
        "↓ marks a per-instance improvement.</p>"
        f'<p><label>Flavour: <select id="runsel">{options}</select></label></p>' + "".join(panels) + "</section>"
    )


TRAJECTORY_ROWS = 80


def _live_card(runs: list[Run]) -> str:
    """Swarm progress from events.jsonl: current generation, agents back, evaluations, cost, recent events."""
    from autoresearch.swarm import format_event

    cards = []
    for run in runs:
        ev = run.events
        if not ev:
            continue
        start = next((e for e in reversed(ev) if e["type"] == "run_start"), ev[0])
        gen = next((e for e in reversed(ev) if e["type"] == "gen_start"), None)
        done = ev[-1]["type"] == "run_end"
        now = ev[-1]["t"] if done else max(ev[-1]["t"], __import__("time").time())
        elapsed = now - start["t"]
        budget = start.get("budget_s", 0)
        back = [e for e in ev if e["type"] == "agent_done" and gen and e["gen"] == gen["gen"] and e["t"] >= gen["t"]]
        stage = "finished" if done else (
            "recording" if ev[-1]["type"] == "entry" else "evaluating on Modal" if ev[-1]["type"] == "eval_start"
            else f"agents working ({len(back)}/{len(gen.get('assignments', [])) if gen else '?'} back)")
        outcomes = {}
        for e in back:
            outcomes[e["outcome"]] = outcomes.get(e["outcome"], 0) + 1
        cost = max(sum(e.cost for e in run.entries),
                   sum(e.get("cost_usd") or 0.0 for e in ev if e["type"] == "agent_done"))
        best = run.best
        tiles = [
            (f"gen {gen['gen'] if gen else '-'}", "generation", stage),
            (f"{int(elapsed // 60)}:{int(elapsed % 60):02d}", "elapsed", f"budget {budget // 60} min" if budget else ""),
            (fmt_num(best.objective) if best else "—", "best objective", f"#{best.id}" if best else ""),
            (str(len(run.proposals)), "proposals recorded",
             "this generation: " + (", ".join(f"{k} {v}" for k, v in outcomes.items()) or "none back yet")),
            (f"${cost:.2f}", "agent cost", "Claude Code reported cost"),
        ]
        recent = "".join(f"<li><code>{escape(format_event(e))}</code></li>" for e in ev[-14:][::-1])
        cards.append(
            f"<h3>{escape(run.label)}{' · LIVE' if not done else ''}</h3><div class='kpis'>"
            + "".join(f"<div class='kpi'><div class='v'>{escape(v)}</div><div class='k'>{escape(k)}</div><div class='d'>{escape(d)}</div></div>"
                      for v, k, d in tiles)
            + f"</div><ul class='events'>{recent}</ul>"
        )
    if not cards:
        return ""
    return "<section class='card'><h2>Now</h2><p class='lead'>Swarm progress from <code>events.jsonl</code> (newest first).</p>" + "".join(cards) + "</section>"


def render_main(runs: list[Run], title: str) -> str:
    colors = {r.label: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    summaries = [summarize(r) for r in runs]
    problem = ", ".join(sorted({r.problem for r in runs}))
    return "\n".join([
        f"<h1>{escape(title)}</h1><p class='sub'>problem: <b>{escape(problem)}</b> · {len(runs)} run{'s' if len(runs) != 1 else ''} · "
        f"updated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
        _live_card(runs),
        _kpis(summaries, runs),
        _progress_card(runs, colors, summaries),
        _scoreboard(summaries, colors, runs),
        _baselines_card(runs),
        _instances_card(runs, colors),
        _outcomes_card(summaries, colors),
        _trajectory_card(runs, colors),
        "<footer>Rendered by <code>python -m autoresearch_viz</code> from "
        + ", ".join(f"<code>{escape(str(r.path))}</code>" for r in runs) + "</footer>",
    ])


LIVE_JS = """
function uiState(){const s={};document.querySelectorAll('.tabs').forEach(t=>{const b=t.querySelector('button.on');if(b)s[t.dataset.group]=b.dataset.key;});
 const sel=document.getElementById('runsel');if(sel)s.__run=sel.value;return s;}
function restore(s){Object.entries(s).forEach(([g,k])=>{if(g==='__run')return;const b=document.querySelector('.tabs[data-group="'+g+'"] button[data-key="'+CSS.escape(k)+'"]');if(b)b.click();});
 const sel=document.getElementById('runsel');if(sel&&s.__run){sel.value=s.__run;sel.dispatchEvent(new Event('change'));}}
async function refresh(){try{const r=await fetch('fragment',{cache:'no-store'});if(!r.ok)return;const html=await r.text();
 const s=uiState(),y=window.scrollY;document.querySelector('main').innerHTML=html;bind();restore(s);window.scrollTo(0,y);}catch(e){}}
setInterval(refresh,3000);
"""


def render(runs: list[Run], title: str = "Autoresearch — objective optimisation & flavour comparison", live: bool = False) -> str:
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{escape(title)}</title><style>{CSS}</style></head><body><main>",
        render_main(runs, title),
        f"</main><div id='tip'></div><script>{JS}{LIVE_JS if live else ''}</script></body></html>",
    ]
    return "\n".join(parts)
