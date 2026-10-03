"""Dependency-free SVG chart primitives (step lines, grouped and stacked horizontal bars)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from itertools import pairwise


def fmt_num(v: float, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    if unit == "tokens":
        return f"{v / 1000:.1f}k" if abs(v) >= 1000 else f"{v:.0f}"
    if unit == "seconds":
        if v >= 3600:
            return f"{v / 3600:.1f}h"
        return f"{v / 60:.1f}m" if v >= 60 else f"{v:.0f}s"
    if unit == "pct":
        return f"{v:+.1f}%"
    if unit == "usd":
        return f"${v:,.2f}"
    return f"{v:,.0f}" if abs(v) >= 100 or float(v).is_integer() else f"{v:.2f}"


def nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.5:
        if t >= lo - step * 0.5:
            ticks.append(round(t, 10))
        t += step
    return ticks


@dataclass
class Series:
    label: str
    color: str
    points: list[tuple[float, float]]
    markers: list[tuple[float, float, str]] = field(default_factory=list)  # (x, y, tooltip)
    end: float | None = None  # where the run really stopped spending; drawn as a dashed vertical line
    switches: list[tuple[float, float, str, str]] = field(default_factory=list)  # (x, y, label, tooltip): model hand-overs


@dataclass
class RefLine:
    y: float
    label: str
    color: str = "#64748b"
    dash: str = "6 4"


def step_chart(
    series: list[Series],
    *,
    refs: list[RefLine],
    x_label: str,
    y_label: str,
    x_unit: str = "",
    width: int = 920,
    height: int = 380,
    extend_to: float | None = None,
) -> str:
    ends = [s for s in series if s.end is not None and s.points]
    ml, mr, mt, mb = 64, 24, 18 + 16 * len(ends), 48  # end-of-run labels get their own band above the plot
    pw, ph = width - ml - mr, height - mt - mb
    xs = [x for s in series for x, _ in s.points] + ([extend_to] if extend_to else [])
    ys = [y for s in series for _, y in s.points] + [r.y for r in refs]
    if not xs or not ys:
        return "<p class='muted'>No scored entries.</p>"
    x_hi = max(xs) or 1
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.08 or 1
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def X(x: float) -> float:
        return ml + pw * x / x_hi

    def Y(y: float) -> float:
        return mt + ph * (1 - (y - y_lo) / (y_hi - y_lo))

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for t in nice_ticks(y_lo, y_hi):
        out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt_num(t)}</text>')
    for t in nice_ticks(0, x_hi, 8):
        if t > x_hi * 1.001:
            continue
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 18}" text-anchor="middle">{fmt_num(t, x_unit)}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/>')
    out.append(f'<text class="label" x="{ml + pw / 2:.0f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<text class="label" transform="translate(14,{mt + ph / 2:.0f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    for r in refs:
        out.append(
            f'<line x1="{ml}" x2="{ml + pw}" y1="{Y(r.y):.1f}" y2="{Y(r.y):.1f}" stroke="{r.color}" stroke-dasharray="{r.dash}" stroke-width="1.5"/>'
        )
        out.append(
            f'<text class="ref" x="{ml + pw - 4}" y="{Y(r.y) - 5:.1f}" text-anchor="end" fill="{r.color}">{escape(r.label)} = {fmt_num(r.y)}</text>'
        )
    for i, s in enumerate(ends):
        x, y, text = X(s.end), 14 + 16 * i, f"{s.label} end · {fmt_num(s.end, x_unit)}"
        out.append(
            f'<g data-run="{escape(s.label, quote=True)}"><line x1="{x:.1f}" x2="{x:.1f}" y1="{y - 10}" y2="{mt + ph}" stroke="{s.color}" '
            f'stroke-dasharray="5 4" stroke-width="1.5" data-tip="{escape(text, quote=True)}"/>'
        )
        anchor, dx = ("end", -6) if x > ml + pw * 0.7 else ("start", 6)
        out.append(f'<text class="end" x="{x + dx:.1f}" y="{y}" text-anchor="{anchor}">{escape(text)}</text></g>')
    for s in series:
        if not s.points:
            continue
        out.append(f'<g data-run="{escape(s.label, quote=True)}">')
        pts = sorted(s.points)
        d = f"M{X(pts[0][0]):.1f},{Y(pts[0][1]):.1f}"
        for (_x0, _y0), (x1, y1) in pairwise(pts):
            d += f" H{X(x1):.1f} V{Y(y1):.1f}"
        last_x = max(extend_to or 0, pts[-1][0])
        d += f" H{X(last_x):.1f}"
        out.append(f'<path d="{d}" fill="none" stroke="{s.color}" stroke-width="2.4" stroke-linejoin="round"/>')
        for x, y, tip in s.markers:
            out.append(
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="5" fill="{s.color}" stroke="#fff" stroke-width="1.5" data-tip="{escape(tip, quote=True)}"/>'
            )
        for x, y, label, tip in s.switches:
            w = 6.2 * len(label) + 10
            bx = min(X(x) + 10, ml + pw - w)
            out.append(
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="8" fill="#fff" stroke="{s.color}" stroke-width="2.5" data-tip="{escape(tip, quote=True)}"/>'
                f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="3" fill="{s.color}" pointer-events="none"/>'
                f'<rect class="swbg" x="{bx:.1f}" y="{Y(y) - 30:.1f}" width="{w:.0f}" height="17" rx="4" stroke="{s.color}"/>'
                f'<text class="end" x="{bx + 5:.1f}" y="{Y(y) - 18:.1f}">{escape(label)}</text>'
            )
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


@dataclass
class BarGroup:
    label: str
    color: str
    values: dict[str, float]  # category -> value
    tips: dict[str, str] = field(default_factory=dict)


def metric_bars(rows: list[tuple[str, float, str]], *, unit: str = "", higher_is_better: bool = True, width: int = 420) -> str:
    """One bar per run from zero, `rows` = [(label, value, color)]; the winner's label and value are bold with a star.

    Each row is a `<g class="mb" data-run>` carrying its value, so the page's run toggles can hide a run and
    re-rank, re-scale and re-stack the others (see `relayoutBars` in html.JS)."""
    vals = [v for _, v, _ in rows if v is not None and math.isfinite(v)]
    if not vals:
        return "<p class='muted'>—</p>"
    best = max(vals) if higher_is_better else min(vals)
    ml, mr, row_h, mt = 92, 96, 26, 4
    pw = width - ml - mr
    hi = max(max(vals), 0) or 1
    out = [f'<svg class="chart mbars" viewBox="0 0 {width} {mt + row_h * len(rows) + 4}" role="img" data-w="{width}" '
           f'data-ml="{ml}" data-pw="{pw}" data-rh="{row_h}" data-higher="{int(higher_is_better)}">',
           f'<line class="axis" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + row_h * len(rows)}"/>']
    for i, (label, v, color) in enumerate(rows):
        ok = v is not None and math.isfinite(v)
        out.append(f'<g class="mb" data-run="{escape(label, quote=True)}" data-v="{v if ok else "nan"}" '
                   f'data-f="{escape(fmt_num(v, unit), quote=True) if ok else ""}" transform="translate(0,{mt + i * row_h})">')
        if not ok:
            out.append(f'<text class="cat" x="{ml - 8}" y="17" text-anchor="end">{escape(label)}</text></g>')
            continue
        win = v == best
        w = max(pw * max(v, 0) / hi, 2)
        weight = ' font-weight="700"' if win else ""
        out.append(f'<text class="cat" x="{ml - 8}" y="17" text-anchor="end"{weight}>{escape(label)}</text>')
        out.append(f'<rect x="{ml}" y="5" width="{w:.1f}" height="15" rx="3" fill="{color}" '
                   f'data-tip="{escape(f"{label}: {fmt_num(v, unit)}", quote=True)}"/>')
        out.append(f'<text class="val" x="{ml + w + 6:.1f}" y="17"{weight}>{fmt_num(v, unit)}{" ★" if win else ""}</text></g>')
    out.append("</svg>")
    return "\n".join(out)


def grouped_hbars(
    categories: list[str],
    groups: list[BarGroup],
    *,
    markers: dict[str, list[tuple[float, str, str]]],
    x_label: str,
    width: int = 920,
    lower_is_better: bool = True,
) -> str:
    """Horizontal grouped bars; `markers[cat]` = [(value, label, color)] drawn as vertical ticks (baseline/optimum)."""
    ml, mr, mt, mb = 190, 24, 12, 44
    bar_h, gap = 14, 3
    group_h = len(groups) * (bar_h + gap) + 14
    height = mt + mb + group_h * len(categories)
    pw = width - ml - mr
    vals = [v for g in groups for v in g.values.values() if math.isfinite(v)] + [m[0] for ms in markers.values() for m in ms]
    x_hi = (max(vals) if vals else 1) * 1.06

    def X(v: float) -> float:
        return ml + pw * v / x_hi

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for t in nice_ticks(0, x_hi, 8):
        if t > x_hi:
            continue
        out.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{height - mb}"/>')
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{height - mb + 16}" text-anchor="middle">{fmt_num(t)}</text>')
    out.append(f'<text class="label" x="{ml + pw / 2:.0f}" y="{height - 6}" text-anchor="middle">{escape(x_label)}</text>')
    for ci, cat in enumerate(categories):
        y0 = mt + ci * group_h + 6
        out.append(f'<text class="cat" x="{ml - 10}" y="{y0 + (group_h - 14) / 2 + 4:.1f}" text-anchor="end">{escape(cat)}</text>')
        if ci:
            out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{y0 - 6}" y2="{y0 - 6}"/>')
        for gi, g in enumerate(groups):
            v = g.values.get(cat)
            if v is None or not math.isfinite(v):
                continue
            y = y0 + gi * (bar_h + gap)
            tip = g.tips.get(cat, f"{g.label}: {fmt_num(v)}")
            out.append(
                f'<g data-run="{escape(g.label, quote=True)}"><rect x="{ml}" y="{y:.1f}" width="{max(X(v) - ml, 1):.1f}" height="{bar_h}" '
                f'fill="{g.color}" rx="2" data-tip="{escape(tip, quote=True)}"/>'
            )
            out.append(f'<text class="val" x="{X(v) + 5:.1f}" y="{y + bar_h - 3:.1f}">{fmt_num(v)}</text></g>')
        for mv, mlabel, mcolor in markers.get(cat, []):
            y_top, y_bot = y0 - 3, y0 + len(groups) * (bar_h + gap) - gap + 3
            out.append(
                f'<line x1="{X(mv):.1f}" x2="{X(mv):.1f}" y1="{y_top:.1f}" y2="{y_bot:.1f}" stroke="{mcolor}" stroke-width="2" stroke-dasharray="4 3" data-tip="{escape(mlabel, quote=True)}: {fmt_num(mv)}"/>'
            )
    out.append("</svg>")
    return "\n".join(out)


def stacked_hbars(rows: list[tuple[str, list[tuple[str, str, float]]]], *, width: int = 920, as_share: bool = False) -> str:
    """rows = [(row label, [(segment label, color, value)])]."""
    ml, mr, mt, mb = 190, 110, 10, 10
    bar_h, gap = 26, 12
    height = mt + mb + len(rows) * (bar_h + gap)
    pw = width - ml - mr
    totals = [sum(v for _, _, v in segs) for _, segs in rows]
    x_hi = 1.0 if as_share else max(totals + [1])
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for ri, ((label, segs), total) in enumerate(zip(rows, totals)):
        y = mt + ri * (bar_h + gap)
        out.append(f'<text class="cat" x="{ml - 10}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{escape(label)}</text>')
        x = ml
        for seg_label, color, v in segs:
            if v <= 0:
                continue
            w = pw * (v / total if as_share else v / x_hi)
            tip = f"{seg_label}: {v:g}" + (f" ({100 * v / total:.0f}%)" if total else "")
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{color}" data-tip="{escape(tip, quote=True)}"/>')
            if w > 22:
                out.append(f'<text class="seg" x="{x + w / 2:.1f}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="middle">{v:g}</text>')
            x += w
        out.append(f'<text class="val" x="{x + 6:.1f}" y="{y + bar_h / 2 + 4:.1f}">{total:g} proposals</text>')
    out.append("</svg>")
    return "\n".join(out)


@dataclass
class DagNode:
    id: int
    layer: int
    tip: str
    colors: dict[str, str]  # colour scheme -> colour; the first scheme is drawn initially
    radius: float = 5.0
    hollow: bool = False  # drawn as an outline (e.g. never evaluated)
    ring: str = ""  # outline colour for highlighted nodes
    label: str = ""


@dataclass
class DagEdge:
    src: int
    dst: int
    color: str = "#94a3b8"
    emphasis: bool = False


def _dag_components(ids: list[int], edges: list[DagEdge]) -> list[list[int]]:
    root = {i: i for i in ids}

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for e in edges:
        root[find(e.dst)] = find(e.src)
    comps: dict[int, list[int]] = {}
    for i in ids:
        comps.setdefault(find(i), []).append(i)
    return sorted(comps.values(), key=lambda c: (-len(c), min(c)))


def _dag_rows(comp: list[int], layer: dict[int, int], parents: dict[int, list[int]], children: dict[int, list[int]]) -> tuple[dict[int, int], int]:
    """Order each layer by the barycentre of its neighbours (a few Sugiyama sweeps), then pack into rows."""
    by_layer: dict[int, list[int]] = {}
    for i in sorted(comp):
        by_layer.setdefault(layer[i], []).append(i)
    layers = sorted(by_layer)
    pos = {i: float(k) for ids in by_layer.values() for k, i in enumerate(ids)}

    def sweep(order: list[int], nbrs: dict[int, list[int]]) -> None:
        for lv in order:
            ids = by_layer[lv]
            key = {i: (sum(pos[n] for n in nbrs[i]) / len(nbrs[i]) if nbrs[i] else pos[i]) for i in ids}
            ids.sort(key=lambda i: (key[i], i))
            pos.update({i: float(k) for k, i in enumerate(ids)})

    for _ in range(4):
        sweep(layers[1:], parents)
        sweep(layers[-2::-1], children)
    height = max(len(v) for v in by_layer.values())
    row: dict[int, int] = {}
    for lv in layers:
        ids = by_layer[lv]
        want = [round(sum(row[p] for p in parents[i]) / len(parents[i])) if parents[i] else k for k, i in enumerate(ids)]
        ys: list[int] = []
        for w in want:
            ys.append(max(w, ys[-1] + 1 if ys else 0))
        for k in range(len(ys) - 1, -1, -1):
            ys[k] = min(ys[k], height - 1 if k == len(ys) - 1 else ys[k + 1] - 1)
        row.update(zip(ids, ys))
    return row, height


def dag_chart(
    nodes: list[DagNode],
    edges: list[DagEdge],
    *,
    layer_label: str = "gen",
    width: int = 1180,
    row_h: int = 20,
    group_label: Callable[[list[int], bool], str] | None = None,
) -> str:
    """Layered DAG, one column per layer, edges drawn left to right. Weakly connected components are
    stacked as separate bands (all single-node components share one band).
    `group_label(ids, is_singles_band)` names a band."""
    if not nodes:
        return "<p class='muted'>No proposals yet.</p>"
    by_id = {n.id: n for n in nodes}
    edges = [e for e in edges if e.src in by_id and e.dst in by_id and by_id[e.src].layer < by_id[e.dst].layer]
    parents: dict[int, list[int]] = {i: [] for i in by_id}
    children: dict[int, list[int]] = {i: [] for i in by_id}
    for e in edges:
        parents[e.dst].append(e.src)
        children[e.src].append(e.dst)
    layer = {n.id: n.layer for n in nodes}
    comps = _dag_components(list(by_id), edges)
    singles = sorted(i for c in comps if len(c) == 1 for i in c)
    bands = [c for c in comps if len(c) > 1] + ([singles] if singles else [])
    layers = sorted(set(layer.values()))
    col = {lv: k for k, lv in enumerate(layers)}
    ml, mr, mt, band_head, band_gap = 14, 14, 26, 20, 12
    cw = max((width - ml - mr) / len(layers), 46.0)
    width = int(ml + mr + cw * len(layers))
    xy: dict[int, tuple[float, float]] = {}
    heads: list[tuple[float, float, str]] = []
    y0 = mt
    for band in bands:
        label = group_label(band, band is singles) if group_label else f"{len(band)} nodes"
        if band is singles:  # isolated nodes need no edges, so pack them into a small grid inside their column
            per_row = max(int(cw // (row_h * 0.8)), 1)
            slot: dict[int, int] = {}
            for i in band:
                k = slot[layer[i]] = slot.get(layer[i], -1) + 1
                gx = (k % per_row - (per_row - 1) / 2) * row_h * 0.8
                xy[i] = (ml + cw * (col[layer[i]] + 0.5) + gx, y0 + band_head + (k // per_row + 0.5) * row_h)
            h = max((k // per_row + 1 for k in slot.values()), default=1)
        else:
            row, h = _dag_rows(band, layer, parents, children)
            for i in band:
                xy[i] = (ml + cw * (col[layer[i]] + 0.5), y0 + band_head + (row[i] + 0.5) * row_h)
        heads.append((y0, y0 + band_head + h * row_h, label))
        y0 += band_head + h * row_h + band_gap
    height = int(y0)
    out = [f'<svg class="lin" viewBox="0 0 {width} {height}" width="{width}" height="{height}" data-w="{width}" data-h="{height}" role="img">']
    for lv in layers:
        x = ml + cw * (col[lv] + 0.5)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{mt - 6}" y2="{height - band_gap}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{mt - 10}" text-anchor="middle">{escape(layer_label)} {lv}</text>')
    for top, _bottom, label in heads:
        if top > mt:
            out.append(f'<line class="axis" x1="{ml}" x2="{width - mr}" y1="{top - band_gap / 2:.1f}" y2="{top - band_gap / 2:.1f}"/>')
        out.append(f'<text class="label" x="{ml}" y="{top + 13:.1f}">{escape(label)}</text>')
    for e in sorted(edges, key=lambda e: e.emphasis):
        (x1, y1), (x2, y2) = xy[e.src], xy[e.dst]
        dx = min(cw * 0.7, (x2 - x1) / 2)
        cls = "le em" if e.emphasis else "le"
        out.append(
            f'<path class="{cls}" data-s="{e.src}" data-t="{e.dst}" stroke="{e.color}" '
            f'd="M{x1:.1f},{y1:.1f} C{x1 + dx:.1f},{y1:.1f} {x2 - dx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
        )
    for n in sorted(nodes, key=lambda n: bool(n.ring)):
        x, y = xy[n.id]
        color = next(iter(n.colors.values()), "#64748b")
        data = " ".join(f'data-c{k}="{v}"' for k, v in n.colors.items())
        paint = f'fill="#fff" stroke="{color}" data-hollow="1"' if n.hollow else f'fill="{color}" stroke="{n.ring or "#fff"}"'
        out.append(
            f'<circle class="ln{" ring" if n.ring else ""}" data-id="{n.id}" cx="{x:.1f}" cy="{y:.1f}" r="{n.radius:.1f}" {paint} {data} '
            f'data-tip="{escape(n.tip, quote=True)}"/>'
        )
        if n.label:
            out.append(f'<text class="nl" x="{x + n.radius + 2:.1f}" y="{y - n.radius:.1f}">{escape(n.label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


@dataclass
class ScatterGroup:
    label: str
    color: str
    points: list[tuple[float, float, str]]  # (x, y, tooltip): one per run


def _mean_sd(vs: list[float]) -> tuple[float, float]:
    vs = [v for v in vs if math.isfinite(v)]
    if not vs:
        return math.nan, math.nan
    m = sum(vs) / len(vs)
    return m, (sum((v - m) ** 2 for v in vs) / (len(vs) - 1)) ** 0.5 if len(vs) > 1 else 0.0


def scatter_chart(
    groups: list[ScatterGroup],
    *,
    x_label: str,
    y_label: str,
    x_unit: str = "",
    y_unit: str = "",
    width: int = 520,
    height: int = 340,
) -> str:
    """Quality (y, higher is better) against a cost (x from zero, lower is better): one faint dot per run,
    a large dot per group at the mean with ±1 sd whiskers, and a dashed line along the groups' Pareto front."""
    ml, mr, mt, mb = 74, 18, 22, 46
    pw, ph = width - ml - mr, height - mt - mb
    xs = [x for g in groups for x, _, _ in g.points if math.isfinite(x)]
    ys = [y for g in groups for _, y, _ in g.points if math.isfinite(y)]
    if not xs or not ys:
        return "<p class='muted'>No finished runs.</p>"
    x_hi = max(xs) * 1.12 or 1
    pad = (max(ys) - min(ys)) * 0.15 or 1
    y_lo, y_hi = min(ys) - pad, max(ys) + pad

    def X(x: float) -> float:
        return ml + pw * x / x_hi

    def Y(y: float) -> float:
        return mt + ph * (1 - (y - y_lo) / (y_hi - y_lo))

    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img">']
    for t in nice_ticks(y_lo, y_hi, 5):
        if y_lo <= t <= y_hi:
            out.append(f'<line class="grid" x1="{ml}" x2="{ml + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
            out.append(f'<text class="tick" x="{ml - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{fmt_num(t, y_unit)}</text>')
    for t in nice_ticks(0, x_hi, 5):
        if t <= x_hi:
            out.append(f'<text class="tick" x="{X(t):.1f}" y="{mt + ph + 17}" text-anchor="middle">{fmt_num(t, x_unit)}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml + pw}" y1="{mt + ph}" y2="{mt + ph}"/>')
    out.append(f'<line class="axis" x1="{ml}" x2="{ml}" y1="{mt}" y2="{mt + ph}"/>')
    out.append(f'<text class="label" x="{ml + pw / 2:.0f}" y="{height - 8}" text-anchor="middle">{escape(x_label)}</text>')
    out.append(f'<text class="label" transform="translate(14,{mt + ph / 2:.0f}) rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
    out.append(f'<text class="tick" x="{ml + 6}" y="{mt + 12}">↖ better (more gain, less spent)</text>')
    means = []
    for g in groups:
        mx, sx = _mean_sd([x for x, _, _ in g.points])
        my, sy = _mean_sd([y for _, y, _ in g.points])
        if math.isfinite(mx) and math.isfinite(my):
            means.append((g, mx, sx, my, sy))
    front, top = [], -math.inf
    for g, mx, _, my, _ in sorted(means, key=lambda m: (m[1], -m[3])):
        if my > top:
            front.append((X(mx), Y(my)))
            top = my
    if len(front) > 1:
        out.append('<polyline class="pareto" points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in front)
                   + '" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="5 4" data-tip="Pareto front of the mean results: '
                   'no configuration is both cheaper and better than a point on it"/>')
    for g, mx, sx, my, sy in means:
        lab = escape(g.label, quote=True)
        out.append(f'<g data-run="{lab}">')
        for x, y, tip in g.points:
            if math.isfinite(x) and math.isfinite(y):
                out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{g.color}" fill-opacity=".4" data-tip="{escape(tip, quote=True)}"/>')
        cx, cy = X(mx), Y(my)
        if sx:
            out.append(f'<line x1="{X(mx - sx):.1f}" x2="{X(mx + sx):.1f}" y1="{cy:.1f}" y2="{cy:.1f}" stroke="{g.color}" stroke-width="1.5"/>')
        if sy:
            out.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(my - sy):.1f}" y2="{Y(my + sy):.1f}" stroke="{g.color}" stroke-width="1.5"/>')
        tip = (f"{g.label}: mean of {len(g.points)} run(s) {fmt_num(my, y_unit)} ± {fmt_num(sy, y_unit).lstrip('+')}, "
               f"{fmt_num(mx, x_unit)} ± {fmt_num(sx, x_unit)}")
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.5" fill="{g.color}" stroke="#fff" stroke-width="2" data-tip="{escape(tip, quote=True)}"/>')
        anchor, dx = ("end", -11) if cx > ml + pw * 0.75 else ("start", 11)
        out.append(f'<text class="end" x="{cx + dx:.1f}" y="{cy - 9:.1f}" text-anchor="{anchor}" font-weight="600">{escape(g.label)}</text></g>')
    out.append("</svg>")
    return "\n".join(out)
