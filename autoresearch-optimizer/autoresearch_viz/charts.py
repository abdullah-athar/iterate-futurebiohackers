"""Dependency-free SVG chart primitives (step lines, grouped and stacked horizontal bars)."""

from __future__ import annotations

import math
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
    ml, mr, mt, mb = 64, 24, 18, 48
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
    for s in series:
        if not s.points:
            continue
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
    out.append("</svg>")
    return "\n".join(out)


@dataclass
class BarGroup:
    label: str
    color: str
    values: dict[str, float]  # category -> value
    tips: dict[str, str] = field(default_factory=dict)


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
                f'<rect x="{ml}" y="{y:.1f}" width="{max(X(v) - ml, 1):.1f}" height="{bar_h}" fill="{g.color}" rx="2" data-tip="{escape(tip, quote=True)}"/>'
            )
            out.append(f'<text class="val" x="{X(v) + 5:.1f}" y="{y + bar_h - 3:.1f}">{fmt_num(v)}</text>')
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
