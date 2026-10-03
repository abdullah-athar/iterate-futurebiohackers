"""Assemble the self-contained dashboard HTML (inline CSS + JS, no external assets)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from html import escape

from .charts import (BarGroup, DagEdge, DagNode, RefLine, Series, dag_chart, fmt_num, grouped_hbars, metric_bars, stacked_hbars,
                     step_chart)
from .load import STATUS_LABELS, STATUS_ORDER, STATUS_REJECTED_DUPLICATE, Run
from .metrics import Summary, best_delta_text, best_so_far, model_switches, per_instance, summarize

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]
STATUS_COLORS = {
    "kept": "#16a34a",
    "evaluated": "#94a3b8",
    "rejected_screen": "#f59e0b",
    "rejected_duplicate": "#fbbf24",
    "failed": "#ef4444",
    "rejected_guard": "#7c2d12",
    "rejected_hypothesis": "#a78bfa",
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
svg .ref{font-size:11px}svg .end{font-size:11px;fill:var(--ink)}svg .swbg{fill:var(--card);stroke-width:1.2}svg .cat{font-size:12px;fill:#0f172a}
svg [data-tip]{cursor:pointer}svg.chart circle[data-tip]:hover{r:7}
.linbar{display:flex;flex-wrap:wrap;gap:6px 22px;align-items:center;margin:0 0 4px}.linbar .tabs{margin:0;align-items:center}.linbar .tabs>span{font-size:13px;color:var(--muted);margin-right:2px}
.linzoom button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:3px 10px;font:inherit;font-size:13px;cursor:pointer;color:var(--muted);margin-left:4px}
.linbox{overflow:auto;max-height:860px;border:1px solid var(--line);border-radius:10px;margin-top:6px}
svg.lin{display:block}svg.lin .le{fill:none;stroke-width:1;opacity:.4}svg.lin .le.em{stroke-width:2.6;opacity:.9}
svg.lin .ln{stroke-width:1}svg.lin .ln[data-hollow]{stroke-width:1.4}svg.lin .ln.ring{stroke-width:2.4}svg.lin .ln:hover{stroke-width:3.5}
svg.lin .nl{font-size:10.5px;fill:#0f172a;font-weight:600;pointer-events:none}
svg.lin.dim .le{opacity:.05}svg.lin.dim .ln,svg.lin.dim .nl{opacity:.2}svg.lin.dim .ln.hl{opacity:1}svg.lin.dim .le.hl{stroke-width:2.2;opacity:.95}
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
.qe{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}.qe-grp{border:1px solid var(--line);border-radius:10px;padding:4px 14px 10px}
.qe-fig{margin:8px 0 0}.qe-prop{margin-right:14px;white-space:nowrap}.qe-fig figcaption{display:flex;justify-content:space-between;gap:8px;font-size:13px}.qe-fig figcaption span{color:var(--muted);font-size:12px}
.runtoggles{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:6px;align-items:center;background:var(--bg);padding:8px 0;margin:0 0 14px}
.runtoggles>span{font-size:13px;color:var(--muted);margin-right:4px}
.runtoggles button{border:1px solid var(--line);background:#fff;border-radius:999px;padding:5px 12px 5px 9px;font:inherit;font-size:13px;cursor:pointer;color:var(--muted)}
.runtoggles button .sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px;opacity:.35}
.runtoggles button.on{color:var(--ink);border-color:#94a3b8}.runtoggles button.on .sw{opacity:1}
.runtoggles button:not(.on){text-decoration:line-through}
.events{margin:0;padding-left:18px;font-size:12.5px;color:#334155}.events li{margin:2px 0}
@media print{.tabs,.runtoggles{display:none}.panel{display:block!important}}
"""

JS = """
const HIDDEN=new Set();
const escHtml=t=>String(t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function relayoutBars(){document.querySelectorAll('svg.mbars').forEach(svg=>{
 const hi=svg.dataset.higher==='1',ml=+svg.dataset.ml,pw=+svg.dataset.pw,rh=+svg.dataset.rh;
 const rows=[...svg.querySelectorAll('g.mb')].filter(g=>!HIDDEN.has(g.dataset.run));
 const vs=rows.map(g=>+g.dataset.v).filter(Number.isFinite);
 const best=vs.length?(hi?Math.max(...vs):Math.min(...vs)):NaN,top=vs.length?(Math.max(...vs,0)||1):1;
 rows.forEach((g,i)=>{g.setAttribute('transform','translate(0,'+(4+i*rh)+')');const v=+g.dataset.v,r=g.querySelector('rect');if(!r)return;
  const w=Math.max(pw*Math.max(v,0)/top,2),t=g.querySelector('.val'),win=v===best;r.setAttribute('width',w.toFixed(1));t.setAttribute('x',(ml+w+6).toFixed(1));
  t.textContent=g.dataset.f+(win?' ★':'');[t,g.querySelector('.cat')].forEach(x=>x.setAttribute('font-weight',win?'700':'400'));});
 svg.setAttribute('viewBox','0 0 '+svg.dataset.w+' '+(8+rh*rows.length));svg.querySelector('.axis').setAttribute('y2',4+rh*rows.length);
 const w=rows.find(g=>+g.dataset.v===best);svg.dataset.winner=w?w.dataset.run:'';});
 document.querySelectorAll('.qe-grp').forEach(grp=>{const svg=grp.querySelector('svg.mbars'),span=grp.querySelector('.qe-win');
  if(span)span.textContent=svg&&svg.dataset.winner?' · '+svg.dataset.winner+' wins':'';});
 document.querySelectorAll('.qe-card').forEach(c=>{const rs=JSON.parse(c.dataset.runs).filter(r=>!HIDDEN.has(r.l)),lead=c.querySelector('.qe-lead');
  if(!rs.length){lead.innerHTML='No run shown';return;}const q=rs.reduce((a,b)=>b.best<a.best?b:a),cs=rs.filter(r=>r.cost);
  let h='<b>'+escHtml(q.l)+'</b> finds the best solver ('+q.bestf+', '+q.gainf+' vs the seed)';
  if(cs.length){const k=cs.reduce((a,b)=>b.cost<a.cost?b:a);h+='; <b>'+escHtml(k.l)+'</b> is the cheapest run ('+k.costf+', '+k.gainf+' vs the seed in '+k.secsf+')';}
  lead.innerHTML=h;});}
function applyRuns(){document.querySelectorAll('[data-run]').forEach(el=>{if(!el.closest('.runtoggles'))el.style.display=HIDDEN.has(el.dataset.run)?'none':'';});
 document.querySelectorAll('.runtoggles button').forEach(b=>{const on=!HIDDEN.has(b.dataset.run);b.classList.toggle('on',on);b.setAttribute('aria-pressed',on);});relayoutBars();}
function bind(){
document.querySelectorAll('.runtoggles button').forEach(b=>b.addEventListener('click',()=>{const r=b.dataset.run;
 if(HIDDEN.has(r))HIDDEN.delete(r);else if(document.querySelectorAll('.runtoggles button.on').length>1)HIDDEN.add(r);applyRuns();}));
applyRuns();
document.querySelectorAll('.tabs').forEach(t=>{t.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
 t.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 const grp=t.dataset.group;document.querySelectorAll('.panel[data-group="'+grp+'"]').forEach(p=>p.classList.toggle('on',p.dataset.key===b.dataset.key));})); });
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{el.addEventListener('mousemove',e=>{tip.textContent=el.dataset.tip;tip.style.display='block';
 tip.style.left=Math.min(e.clientX+14,window.innerWidth-380)+'px';tip.style.top=(e.clientY+14)+'px';});el.addEventListener('mouseleave',()=>tip.style.display='none');});
const sel=document.getElementById('runsel');if(sel){sel.addEventListener('change',()=>{document.querySelectorAll('.traj').forEach(p=>p.classList.toggle('on',p.dataset.key===sel.value));});}
document.querySelectorAll('.lincolor button').forEach(b=>b.addEventListener('click',()=>{b.closest('.linrun').querySelectorAll('.ln').forEach(n=>{
 n.setAttribute(n.dataset.hollow?'stroke':'fill',n.dataset['c'+b.dataset.key]);});}));
document.querySelectorAll('.linzoom button').forEach(b=>b.addEventListener('click',()=>{const box=b.closest('.linrun');
 const z=b.dataset.z==='1'?1:Math.max(.3,Math.min(5,(+box.dataset.z||1)*(+b.dataset.z)));box.dataset.z=z;
 box.querySelectorAll('svg.lin').forEach(s=>{s.setAttribute('width',s.dataset.w*z);s.setAttribute('height',s.dataset.h*z);});}));
document.querySelectorAll('svg.lin').forEach(svg=>{const up={},dn={},nodes={};
 svg.querySelectorAll('.le').forEach(p=>{(up[p.dataset.t]=up[p.dataset.t]||[]).push(p);(dn[p.dataset.s]=dn[p.dataset.s]||[]).push(p);});
 svg.querySelectorAll('.ln').forEach(n=>nodes[n.dataset.id]=n);
 const walk=(id,m,k,seen)=>{(m[id]||[]).forEach(p=>{p.classList.add('hl');const o=p.dataset[k];if(!seen.has(o)){seen.add(o);if(nodes[o])nodes[o].classList.add('hl');walk(o,m,k,seen);}});};
 svg.querySelectorAll('.ln').forEach(n=>{n.addEventListener('mouseenter',()=>{svg.classList.add('dim');n.classList.add('hl');walk(n.dataset.id,up,'s',new Set());walk(n.dataset.id,dn,'t',new Set());});
  n.addEventListener('mouseleave',()=>{svg.classList.remove('dim');svg.querySelectorAll('.hl').forEach(x=>x.classList.remove('hl'));});});});
}
bind();
"""


def _badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{escape(text)}</span>'


def _legend(items: list[tuple[str, str]], runs: frozenset[str] = frozenset()) -> str:
    """Colour key; items whose label is in `runs` follow the page's run toggles."""
    def tag(label: str) -> str:
        return f' data-run="{escape(label, quote=True)}"' if label in runs else ""

    return (
        '<div class="legend">'
        + "".join(f'<span{tag(l)}><span class="sw" style="background:{c}"></span>{escape(l)}</span>' for l, c in items)
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
    if any(e.cost for r in runs for e in r.entries):
        axes.append(("cost", "cumulative agent cost (Claude Code's estimate)", "usd"))
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
            end = max((x for x, _ in xy), default=None) if key != "evals" and len(runs) > 1 else None
            switches = []
            for p, old, new in model_switches(run):
                switches.append((getattr(p, key), p.best, f"{_model_name(old)} → {_model_name(new)}",
                                 f"{run.label} switches from {_model_name(old)} to {_model_name(new)} after #{p.entry_id}: "
                                 f"{fmt_num(p.seconds, 'seconds')} into the run, {fmt_num(p.cost, 'usd')} spent, "
                                 f"{fmt_num(p.tokens, 'tokens')} tokens, best so far {fmt_num(p.best)}"))
            series.append(Series(run.label, colors[run.label], xy, marks, end, switches))
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
        '<section class="card"><h2>Research progress</h2><p class="lead">Best objective found so far. Dots mark proposals that set a new global best; hover for the hypothesis.'
        + (" A ring marks where a run hands over to another model (<code>--model sonnet,opus</code>); hover it for the time and cost spent at that moment."
           if any(model_switches(r) for r in runs) else "")
        + "</p>"
        f'<div class="tabs" data-group="x">{"".join(buttons)}</div>'
        + _legend([(r.label, colors[r.label]) for r in runs], frozenset(r.label for r in runs))
        + "".join(panels)
        + "</section>"
    )


def _quality_efficiency_card(summaries: list[Summary], colors: dict[str, str], runs: list[Run]) -> str:
    """Two or more runs (e.g. one per model): what each one found versus what it cost, with the winner of each measure."""
    pairs = [(s, r) for s, r in zip(summaries, runs) if math.isfinite(s.best_objective)]
    if len(pairs) < 2:
        return ""

    def held_out_gain(s: Summary, r: Run) -> float:
        seed = r.seed.evals.get("holdout") if r.seed else None
        if s.holdout_score is None or not seed or not seed.ok or not seed.score:
            return math.nan
        return 100.0 * (seed.score - s.holdout_score) / seed.score

    def per_dollar(s: Summary) -> float:
        return (s.seed_objective - s.best_objective) / s.cost_usd if s.cost_usd else math.nan

    def block(title: str, note: str, value, unit: str, higher: bool) -> str:
        rows = [(s.label, value(s, r), colors[s.label]) for s, r in pairs]
        return (f"<figure class='qe-fig'><figcaption><b>{escape(title)}</b><span>{escape(note)}</span></figcaption>"
                + metric_bars(rows, unit=unit, higher_is_better=higher) + "</figure>")

    quality = [("gain over the seed", f"{pairs[0][0].objective_split} split · higher is better",
                lambda s, r: s.gain_vs_seed_pct, "pct", True),
               ("held-out gain over the seed", "instances the search never saw · higher is better", held_out_gain, "pct", True)]
    efficiency = [("total agent cost", "Claude Code's estimate · lower is better", lambda s, r: s.cost_usd or math.nan, "usd", False),
                  ("wall-clock", "first to last ledger entry · lower is better", lambda s, r: s.seconds, "seconds", False),
                  ("objective points gained per dollar", "higher is better", lambda s, r: per_dollar(s), "", True)]
    best_q = min(pairs, key=lambda p: p[0].best_objective)[0]
    costed = [p for p in pairs if p[0].cost_usd]
    cheap = min(costed, key=lambda p: p[0].cost_usd)[0] if costed else None
    lead = (f"<b>{escape(best_q.label)}</b> finds the best solver ({fmt_num(best_q.best_objective)}, "
            f"{fmt_num(best_q.gain_vs_seed_pct, 'pct')} vs the seed)")
    if cheap:
        lead += (f"; <b>{escape(cheap.label)}</b> is the cheapest run ({fmt_num(cheap.cost_usd, 'usd')}, "
                 f"{fmt_num(cheap.gain_vs_seed_pct, 'pct')} vs the seed in {fmt_num(cheap.seconds, 'seconds')})")
    models = "".join(f"<span class='qe-prop' data-run='{escape(s.label, quote=True)}'>{escape(s.label)}: <code>{escape(_proposers(r))}</code></span>"
                     for s, r in pairs)
    data = [{"l": s.label, "best": s.best_objective, "bestf": fmt_num(s.best_objective), "gainf": fmt_num(s.gain_vs_seed_pct, "pct"),
             "cost": s.cost_usd or 0, "costf": fmt_num(s.cost_usd, "usd"), "secsf": fmt_num(s.seconds, "seconds")} for s, _ in pairs]
    return (
        f"<section class='card qe-card' data-runs='{escape(json.dumps(data), quote=True)}'><h2>Quality vs efficiency</h2>"
        f"<p class='lead'><span class='qe-lead'>{lead}</span>. A ★ marks the winner of each measure among the runs shown. Proposers: {models}.</p><div class='qe'>"
        f"<div class='qe-grp'><h3>Quality<span class='qe-win'> · {escape(best_q.label)} wins</span></h3>"
        + "".join(block(*q) for q in quality)
        + f"</div><div class='qe-grp'><h3>Efficiency<span class='qe-win'>{(' · ' + escape(cheap.label) + ' wins') if cheap else ''}</span></h3>"
        + "".join(block(*e) for e in efficiency)
        + "</div></div><p class='muted' style='font-size:12.5px;margin:10px 0 0'>The 'vs cost' tab of <i>Research progress</i> "
        "shows the same trade-off over the run. With one run per flavour, differences of a few percent are within noise.</p></section>"
    )


def _model_name(proposer: str) -> str:
    """'claude-code:sonnet' -> 'Sonnet'."""
    name = proposer.split(":", 1)[-1]
    return name[:1].upper() + name[1:]


def _proposers(run: Run) -> str:
    """The run's proposers in first-use order, e.g. 'claude-code:sonnet → claude-code:opus' for a model schedule."""
    return " → ".join(dict.fromkeys(e.proposer for e in run.proposals)) or "?"


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
            f"<tr class='{'best' if s is best_label else ''}' data-run='{escape(s.label, quote=True)}'><td class='l'><span class='sw' style='background:{colors[s.label]}'></span><b>{escape(s.label)}</b>"
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
            f"<tr data-run='{escape(s.label, quote=True)}'><td class='l'><span class='sw' style='background:{colors[s.label]}'></span>{escape(s.label)}</td>{''.join(cells)}</tr>"
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
        [(r.label, colors[r.label]) for r in runs] + [("baseline (dashed grey)", "#64748b"), ("planted reference (dashed green; a feasible answer, not a proven optimum)", "#16a34a")],
        frozenset(r.label for r in runs),
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

LINEAGE_VIEWS = [("ideas", "ideas (seed hidden)"), ("evaluated", "evaluated only"), ("seed", "with seed")]
LINEAGE_SCHEMES = [("status", "outcome"), ("verdict", "verdict"), ("mode", "mode")]
BEST_PATH_COLOR = "#dc2626"


def _lineage_layers(run: Run) -> dict[int, int]:
    """Column per proposal: its swarm generation, pushed right of its parents (tree depth when there is no generation)."""
    by_id = {e.id: e for e in run.entries}
    layer: dict[int, int] = {}

    def lay(e, stack: frozenset = frozenset()) -> int:
        if e.id not in layer:
            ps = [by_id[p] for p in e.parent_ids if p in by_id and p not in stack]
            depth = 1 + max((lay(p, stack | {e.id}) for p in ps), default=-1)
            layer[e.id] = max(e.generation if isinstance(e.generation, int) else depth, depth)
        return layer[e.id]

    for e in run.entries:
        lay(e)
    return layer


def _lineage_svg(run: Run, view: str, layer: dict[int, int]) -> str:
    by_id = {e.id: e for e in run.entries}
    shown = [e for e in run.entries if (view == "seed" or not e.is_seed) and (view != "evaluated" or e.status != STATUS_REJECTED_DUPLICATE)]
    shown_ids = {e.id for e in shown}
    n_children = {e.id: 0 for e in run.entries}
    dup_children = dict(n_children)
    for e in run.entries:
        for p in e.parent_ids:
            if p in n_children:
                n_children[p] += 1
                dup_children[p] += e.status == STATUS_REJECTED_DUPLICATE
    best = run.best
    best_path: set[int] = set()
    stack = [best.id] if best else []
    while stack:
        i = stack.pop()
        if i in by_id and i not in best_path:
            best_path.add(i)
            stack.extend(by_id[i].parent_ids)
    ranked = sorted((e for e in run.entries if e.scored), key=lambda e: (e.objective, e.id))
    rank = {e.id: k / max(len(ranked) - 1, 1) for k, e in enumerate(ranked)}
    nodes = []
    for e in shown:
        scored_parents = [by_id[p].objective for p in e.parent_ids if p in by_id and by_id[p].scored]
        is_best = best is not None and e.id == best.id
        lines = [f"#{e.id} · {e.mode or '?'}" + (f" · gen {e.generation}" if e.generation is not None else "")]
        if e.scored:
            lines.append(
                f"objective {fmt_num(e.objective)}"
                + (f" ({e.objective - min(scored_parents):+,.0f} vs best parent)" if scored_parents else "")
                + (" · FINAL BEST" if is_best else " · new global best" if e.improved_global and not e.is_seed else "")
            )
        elif e.status == STATUS_REJECTED_DUPLICATE:
            dup = e.novelty.get("duplicate_of")
            lines.append("not evaluated: near-duplicate" + (f" of #{dup}" if dup is not None else ""))
        lines.append(STATUS_LABELS.get(e.status, e.status) + (f" · verdict {e.verdict}" if e.verdict else ""))
        lines.append(
            ("parents " + ", ".join(f"#{p}" for p in e.parent_ids) if e.parent_ids else "root")
            + f" · {n_children[e.id]} follow-up{'s' if n_children[e.id] != 1 else ''}"
            + (f" ({dup_children[e.id]} duplicates hidden)" if view == "evaluated" and dup_children[e.id] else "")
        )
        lines.append(e.hypothesis)
        global_best = e.improved_global and not e.is_seed
        nodes.append(DagNode(
            id=e.id,
            layer=layer[e.id],
            tip="\n".join(lines),
            colors={
                "status": STATUS_COLORS.get(e.status, "#64748b"),
                "verdict": VERDICT_COLORS.get(e.verdict, STATUS_COLORS["seed"] if e.is_seed else "#64748b"),
                "mode": MODE_COLORS.get(e.mode, "#64748b"),
            },
            radius=3.5 + 5.0 * (1 - rank[e.id]) if e.scored else 3.2,
            hollow=not e.scored,
            ring=BEST_PATH_COLOR if e.id in best_path and (global_best or is_best) else "#0f172a" if global_best else "",
            label=f"★ #{e.id} best" if is_best else f"#{e.id}" if global_best else "",
        ))
    edges = [
        DagEdge(p, e.id, BEST_PATH_COLOR if (p in best_path and e.id in best_path) else MODE_COLORS["merge"] if len(e.parent_ids) > 1 else "#94a3b8",
                emphasis=p in best_path and e.id in best_path)
        for e in shown for p in e.parent_ids if p in shown_ids
    ]

    def group_label(ids: list[int], singles: bool) -> str:
        if singles:
            return f"{len(ids)} one-off idea{'s' if len(ids) != 1 else ''} with no follow-ups"
        roots = sorted(i for i in ids if not any(p in shown_ids for p in by_id[i].parent_ids))
        scored = [by_id[i] for i in ids if by_id[i].scored and by_id[i].confirmed is not False]
        top = min(scored, key=lambda e: (e.objective, e.id)) if scored else None
        return (
            f"{len(ids)} ideas from " + ", ".join(f"#{r}" for r in roots[:8]) + (" …" if len(roots) > 8 else "")
            + (f" · best {fmt_num(top.objective)} (#{top.id})" if top else "")
            + (" · contains the final best" if best and best.id in ids else "")
        )

    return dag_chart(nodes, edges, layer_label="gen" if any(e.generation is not None for e in run.entries) else "depth", group_label=group_label)


def _lineage_card(runs: list[Run]) -> str:
    schemes = {"status": STATUS_COLORS, "verdict": VERDICT_COLORS, "mode": MODE_COLORS}
    blocks = []
    for i, run in enumerate(runs):
        layer = _lineage_layers(run)
        present = {k: {getattr(e, k) for e in run.entries} for k in schemes}
        legends = "".join(
            f'<div class="panel {"on" if k == "status" else ""}" data-group="lincolor{i}" data-key="{k}">'
            + _legend([(STATUS_LABELS.get(v, v) if k == "status" else v, c) for v, c in schemes[k].items() if v in present[k]])
            + "</div>"
            for k, _ in LINEAGE_SCHEMES
        )
        views = "".join(
            f'<div class="panel {"on" if j == 0 else ""}" data-group="linview{i}" data-key="{k}">{_lineage_svg(run, k, layer)}</div>'
            for j, (k, _) in enumerate(LINEAGE_VIEWS)
        )
        bar = (
            f'<div class="linbar"><div class="tabs" data-group="linview{i}">'
            + "".join(f'<button class="{"on" if j == 0 else ""}" data-key="{k}">{escape(t)}</button>' for j, (k, t) in enumerate(LINEAGE_VIEWS))
            + f'</div><div class="tabs lincolor" data-group="lincolor{i}"><span>colour by</span>'
            + "".join(f'<button class="{"on" if j == 0 else ""}" data-key="{k}">{escape(t)}</button>' for j, (k, t) in enumerate(LINEAGE_SCHEMES))
            + '</div><div class="linzoom"><span class="muted" style="font-size:13px">zoom</span>'
            '<button data-z="0.8">−</button><button data-z="1.25">+</button><button data-z="1">reset</button></div></div>'
        )
        blocks.append(
            f'<div class="linrun panel {"on" if i == 0 else ""}" data-group="linrun" data-key="{escape(run.label, quote=True)}">'
            + bar + legends + _LINEAGE_KEY + f'<div class="linbox">{views}</div></div>'
        )
    run_tabs = (
        '<div class="tabs" data-group="linrun">'
        + "".join(f'<button class="{"on" if i == 0 else ""}" data-key="{escape(r.label, quote=True)}">{escape(r.label)}</button>' for i, r in enumerate(runs))
        + "</div>"
    ) if len(runs) > 1 else ""
    return (
        "<section class='card'><h2>Idea lineage</h2><p class='lead'>Every proposal as a node, with an edge from each parent it was "
        "built on (merges have several). Columns are generations. The seed is hidden by default, so ideas that started "
        "independently from it form separate trees. Hover a node for its hypothesis and to trace its ancestors and descendants.</p>"
        + run_tabs + "".join(blocks) + "</section>"
    )


def _key_icon(svg: str, text: str) -> str:
    return f'<span><svg width="22" height="14" style="vertical-align:-3px;margin-right:6px">{svg}</svg>{escape(text)}</span>'


_LINEAGE_KEY = (
    '<div class="legend">'
    + _key_icon('<circle cx="5" cy="7" r="3" fill="#94a3b8"/><circle cx="15" cy="7" r="6.5" fill="#94a3b8"/>', "size = objective rank (bigger is better)")
    + _key_icon('<circle cx="11" cy="7" r="4" fill="#fff" stroke="#fbbf24" stroke-width="1.5"/>', "hollow = not evaluated")
    + _key_icon('<circle cx="11" cy="7" r="5" fill="#16a34a" stroke="#0f172a" stroke-width="2.2"/>', "new global best")
    + _key_icon(f'<line x1="0" y1="7" x2="22" y2="7" stroke="{BEST_PATH_COLOR}" stroke-width="3"/>', "path to the final best (★)")
    + _key_icon(f'<line x1="0" y1="7" x2="22" y2="7" stroke="{MODE_COLORS["merge"]}" stroke-width="2"/>', "edge into a merge")
    + "</div>"
)


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
            f"<div data-run='{escape(run.label, quote=True)}'><h3>{escape(run.label)}{' · LIVE' if not done else ''}</h3><div class='kpis'>"
            + "".join(f"<div class='kpi'><div class='v'>{escape(v)}</div><div class='k'>{escape(k)}</div><div class='d'>{escape(d)}</div></div>"
                      for v, k, d in tiles)
            + f"</div><ul class='events'>{recent}</ul></div>"
        )
    if not cards:
        return ""
    return "<section class='card'><h2>Now</h2><p class='lead'>Swarm progress from <code>events.jsonl</code> (newest first).</p>" + "".join(cards) + "</section>"


def _run_toggles(runs: list[Run], colors: dict[str, str]) -> str:
    """One button per run: hides or shows that run in every chart, legend and table (at least one stays shown)."""
    if len(runs) < 2:
        return ""
    return ("<div class='runtoggles' role='group' aria-label='Runs shown'><span>Show:</span>"
            + "".join(f"<button class='on' aria-pressed='true' data-run='{escape(r.label, quote=True)}'>"
                      f"<span class='sw' style='background:{colors[r.label]}'></span>{escape(r.label)}</button>" for r in runs)
            + "</div>")


def render_main(runs: list[Run], title: str) -> str:
    colors = {r.label: PALETTE[i % len(PALETTE)] for i, r in enumerate(runs)}
    summaries = [summarize(r) for r in runs]
    problem = ", ".join(sorted({r.problem for r in runs}))
    return "\n".join([
        f"<h1>{escape(title)}</h1><p class='sub'>problem: <b>{escape(problem)}</b> · {len(runs)} run{'s' if len(runs) != 1 else ''} · "
        f"updated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
        _run_toggles(runs, colors),
        _live_card(runs),
        _kpis(summaries, runs),
        _quality_efficiency_card(summaries, colors, runs),
        _progress_card(runs, colors, summaries),
        _scoreboard(summaries, colors, runs),
        _baselines_card(runs),
        _instances_card(runs, colors),
        _outcomes_card(summaries, colors),
        _lineage_card(runs),
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
