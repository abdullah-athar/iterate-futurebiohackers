#!/usr/bin/env python3
"""Leaderboard and Pareto plot for the speedrun A/B registry (standard library only).

Inputs (all optional; defaults live under artifacts/speedrun_runs/):
  registry.jsonl    one JSON object per launcher run row (controls and variants)
  k.json            {"k": <pp per second>, "source": str, "time": iso}: the exchange rate
  gpu_ledger.jsonl  one JSON object per container: time, label, gpu, power, gpu_minutes, note

Outputs (into --out-dir, default artifacts/speedrun_runs/): leaderboard.md and pareto.svg.
The variant table is also printed to stdout; --json prints the sorted rows as JSON instead.

    uv run python scripts/leaderboard.py [--registry P] [--ledger P] [--k-file P] [--out-dir D]
                                         [--budget-min N] [--json] [--no-write]

dtime_adj = dtime - dacc_pp / k (seconds, lower is better) is recomputed from the CURRENT k for
every row that has paired data: k from k.json overrides the row's k_used; a row falls back to
its own k_used only when k.json is missing or invalid; with neither the cell stays blank.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "speedrun_runs"
DEFAULT_BUDGET_MIN = 1263
HYPOTHESIS_CHARS = 60
FLAG = "!"

GUARD_MISS_RE = re.compile(r"guard miss|pcie-guard attempt", re.IGNORECASE)
OK_VERDICT_RE = re.compile(r"\b(qualified|complete)\b", re.IGNORECASE)
BAD_VERDICT_RE = re.compile(
    r"incomplete|below|fail|no summary|skipped|error|interrupted", re.IGNORECASE
)
CONFIRMED_RE = re.compile(r"(?<!un)confirmed", re.IGNORECASE)
NOT_CONFIRMED_RE = re.compile(r"not\s+confirmed", re.IGNORECASE)

VARIANT_COLUMNS = [
    "rank",
    "label",
    "round",
    "params",
    "hypothesis",
    "n",
    "mean acc %",
    "acc std",
    "dacc pp +- SE",
    "dtime s",
    "dtime_adj s",
    "mean time s",
    "build s",
    "nonfinite",
    "GPU @ power",
    "verdict",
]
CONTROL_COLUMNS = [
    "job",
    "label",
    "round",
    "GPU @ power",
    "n",
    "mean acc %",
    "acc std",
    "mean time s",
    "time std",
    "build s",
    "verdict",
]
COMPUTED_KEYS = [
    "rank",
    "is_control",
    "dtime_adj",
    "dtime_adj_se",
    "dtime_adj_registry",
    "k_effective",
    "k_source",
    "dacc_pp",
    "dacc_se_pp",
    "mean_acc_pct",
    "acc_std_pct",
    "params_compact",
    "gpu_at_power",
    "flags",
    "flagged",
]

# Chart chrome and the validated categorical palette (dataviz skill reference instance).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


# ----------------------------------------------------------------------------- small helpers
def warn(msg: str) -> None:
    print(f"leaderboard: warning: {msg}", file=sys.stderr)


def rel(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else the absolute POSIX path."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def num(x) -> float | None:
    """float(x) for a finite number or numeric string, else None (bools are not numbers)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x) if isinstance(x, (int, float)) else float(str(x).strip())
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def pct(x: float | None) -> float | None:
    return None if x is None else x * 100.0


def fmt(x: float | None, nd: int = 2, sign: bool = False) -> str:
    if x is None:
        return ""
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def trunc(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def natural_key(s) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def power_watts(p) -> float | None:
    if isinstance(p, (int, float)) and not isinstance(p, bool):
        return float(p) if math.isfinite(p) else None
    m = re.search(r"-?\d+(?:\.\d+)?", str(p or ""))
    return float(m.group()) if m else None


def fmt_power(p) -> str:
    w = power_watts(p)
    if w is None:
        return "?" if p in (None, "", "?") else str(p)
    return f"{w:.0f} W"


def short_gpu(name) -> str:
    s = str(name or "?").replace("NVIDIA ", "")
    s = s.replace("-80GB", "").replace(" 80GB", "").replace("-SXM4", " SXM4")
    return re.sub(r"\s+", " ", s).strip() or "?"


def gpu_at_power(gpu, power) -> str:
    return f"{short_gpu(gpu)} @ {fmt_power(power)}"


def params_compact(p) -> str:
    """Compact JSON for a params object; the launcher may also store it as a JSON string."""
    if isinstance(p, str):
        try:
            p = json.loads(p) if p.strip() else {}
        except json.JSONDecodeError:
            return p.strip()
    if p is None:
        p = {}
    return json.dumps(p, separators=(",", ":"))


def verdict_ok(verdict) -> bool:
    v = str(verdict or "")
    return bool(OK_VERDICT_RE.search(v)) and not BAD_VERDICT_RE.search(v)


def verdict_confirmed(verdict) -> bool:
    v = str(verdict or "")
    return bool(CONFIRMED_RE.search(v)) and not NOT_CONFIRMED_RE.search(v)


def mean_of(row: dict, key: str, list_key: str) -> float | None:
    v = num(row.get(key))
    if v is None and isinstance(row.get(list_key), list):
        xs = [y for y in (num(x) for x in row[list_key]) if y is not None]
        v = statistics.fmean(xs) if xs else None
    return v


# ----------------------------------------------------------------------------- inputs
def read_jsonl(path: Path, what: str) -> list[dict]:
    if not path.exists():
        warn(f"{what} not found: {path}")
        return []
    rows: list[dict] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            warn(f"{what} line {i}: not JSON ({exc.msg}); skipped")
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            warn(f"{what} line {i}: not a JSON object; skipped")
    return rows


def load_k(path: Path) -> tuple[float | None, dict]:
    """Return (k in pp per second or None, the parsed k.json contents or {})."""
    if not path.exists():
        warn(f"k file not found ({path}): dtime_adj falls back to each row's k_used")
        return None, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"k file unreadable ({exc}): falling back to each row's k_used")
        return None, {}
    k = num(data.get("k")) if isinstance(data, dict) else None
    if k is None or k <= 0:
        warn(f"k file {path} has no positive finite 'k': falling back to each row's k_used")
        return None, data if isinstance(data, dict) else {}
    return k, data


def is_control(row: dict) -> bool:
    v = row.get("is_control")
    if v is None:
        return str(row.get("label") or "").strip().lower() == "control"
    return bool(v)


ROUND_K: dict[str, float] = {}  # rounds measured on an earlier base keep that base's k


def build_entry(row: dict, k_current: float | None) -> dict:
    """Normalise one registry row and recompute dtime_adj from the current k (or the k of the
    row's round when k.json lists it under "previous")."""
    paired = row.get("paired") if isinstance(row.get("paired"), dict) else None

    def pv(key: str) -> float | None:
        return num(paired.get(key)) if paired else None

    dacc, dacc_se, dtime, dtime_se = pv("dacc"), pv("dacc_se"), pv("dtime"), pv("dtime_se")
    k_used = num(row.get("k_used"))
    k_round = ROUND_K.get(str(row.get("round")))
    if k_round is not None:
        k_eff, k_src = k_round, "k.json(previous)"
    elif k_current is not None:
        k_eff, k_src = k_current, "k.json"
    elif k_used is not None and k_used > 0:
        k_eff, k_src = k_used, "k_used"
    else:
        k_eff, k_src = None, None
    adj = adj_se = None
    if dacc is not None and dtime is not None and k_eff is not None:
        adj = dtime - dacc * 100.0 / k_eff
        terms = [dtime_se, None if dacc_se is None else dacc_se * 100.0 / k_eff]
        terms = [t for t in terms if t is not None]
        adj_se = math.sqrt(sum(t * t for t in terms)) if terms else None
    nonfinite = num(row.get("nonfinite"))
    flags: list[str] = []
    if nonfinite is not None and nonfinite > 0:
        flags.append(f"nonfinite={int(nonfinite)}")
    if not verdict_ok(row.get("verdict")):
        flags.append(f"verdict: {row.get('verdict') or 'missing'}")
    return {
        "raw": row,
        "rank": None,
        "is_control": is_control(row),
        "label": str(row.get("label") or "?"),
        "round": "" if row.get("round") is None else str(row.get("round")),
        "job": "" if row.get("job") is None else str(row.get("job")),
        "time": str(row.get("time") or ""),
        "params_compact": params_compact(row.get("params")),
        "hypothesis": str(row.get("hypothesis") or ""),
        "n": num(row.get("n")),
        "trials_ok": num(row.get("trials_ok")),
        "pairs": pv("pairs"),
        "mean_acc_pct": pct(mean_of(row, "mean_acc", "accs")),
        "acc_std_pct": pct(num(row.get("acc_std"))),
        "dacc_pp": pct(dacc),
        "dacc_se_pp": pct(dacc_se),
        "dtime": dtime,
        "dtime_se": dtime_se,
        "dtime_adj": adj,
        "dtime_adj_se": adj_se,
        "dtime_adj_registry": num(row.get("dtime_adj")),
        "k_effective": k_eff,
        "k_source": k_src,
        "mean_time": mean_of(row, "mean_time", "times"),
        "time_std": num(row.get("time_std")),
        "build_time": num(row.get("build_time")),
        "build_mode": str(row.get("build_mode") or ""),
        "nonfinite": nonfinite,
        "gpu_at_power": gpu_at_power(row.get("gpu"), row.get("power_limit")),
        "verdict": str(row.get("verdict") or ""),
        "flags": flags,
        "flagged": bool(flags),
    }


def sort_key(e: dict) -> tuple:
    if e["dtime_adj"] is not None:
        return (0, e["dtime_adj"], 0.0)
    dtime = math.inf if e["dtime"] is None else e["dtime"]
    mean_time = math.inf if e["mean_time"] is None else e["mean_time"]
    return (1, dtime, mean_time)


def ledger_stats(entries: list[dict]) -> dict:
    groups: dict[tuple[str, str], dict] = {}
    total = misses = miss_minutes = 0.0
    for e in entries:
        minutes = num(e.get("gpu_minutes")) or 0.0
        miss = bool(GUARD_MISS_RE.search(str(e.get("label") or "")))
        key = (short_gpu(e.get("gpu")), fmt_power(e.get("power")))
        g = groups.setdefault(key, {"containers": 0, "misses": 0, "minutes": 0.0})
        g["containers"] += 1
        g["minutes"] += minutes
        g["misses"] += miss
        total += minutes
        misses += miss
        miss_minutes += minutes if miss else 0.0
    rows = sorted(groups.items(), key=lambda kv: (-kv[1]["containers"], kv[0]))
    n = len(entries)
    return {
        "containers": n,
        "total_minutes": total,
        "guard_misses": int(misses),
        "guard_miss_rate": (misses / n) if n else None,
        "guard_miss_minutes": miss_minutes,
        "groups": [
            {
                "gpu": gpu,
                "power": power,
                "containers": g["containers"],
                "share": g["containers"] / n if n else None,
                "guard_misses": int(g["misses"]),
                "minutes": g["minutes"],
            }
            for (gpu, power), g in rows
        ],
    }


# ----------------------------------------------------------------------------- tables
def fmt_n(e: dict) -> str:
    n, ok = e["n"], e["trials_ok"]
    if ok is not None and n is not None:
        return f"{int(ok)}/{int(n)}"
    if n is not None:
        return str(int(n))
    return "" if ok is None else f"{int(ok)}/?"


def fmt_build(e: dict) -> str:
    if e["build_time"] is None:
        return e["build_mode"]
    return (f"{e['build_time']:.0f} {e['build_mode']}").strip()


def variant_cells(e: dict, compact: bool) -> list:
    dacc = ""
    if e["dacc_pp"] is not None:
        dacc = f"{e['dacc_pp']:+.2f}"
        if e["dacc_se_pp"] is not None:
            dacc += f" +- {e['dacc_se_pp']:.2f}"
    adj = ""
    if e["dtime_adj"] is not None:
        adj = f"{e['dtime_adj']:+.2f}" + ("*" if e["k_source"] == "k_used" else "")
    label = f"{FLAG} {e['label']}" if e["flagged"] else e["label"]
    params = e["params_compact"] if compact else f"`{e['params_compact']}`"
    hypothesis = trunc(e["hypothesis"], 32 if compact else HYPOTHESIS_CHARS)
    verdict = trunc(e["verdict"], 30) if compact else e["verdict"]
    return [
        e["rank"],
        label,
        e["round"],
        params,
        hypothesis,
        fmt_n(e),
        fmt(e["mean_acc_pct"]),
        fmt(e["acc_std_pct"]),
        dacc,
        fmt(e["dtime"], sign=True),
        adj,
        fmt(e["mean_time"]),
        fmt_build(e),
        "" if e["nonfinite"] is None else str(int(e["nonfinite"])),
        e["gpu_at_power"],
        verdict,
    ]


def control_cells(e: dict) -> list:
    return [
        e["job"],
        e["label"],
        e["round"],
        e["gpu_at_power"],
        fmt_n(e),
        fmt(e["mean_acc_pct"]),
        fmt(e["acc_std_pct"]),
        fmt(e["mean_time"]),
        fmt(e["time_std"]),
        fmt_build(e),
        e["verdict"],
    ]


def hit_rate_cells(stats: dict) -> list[list]:
    return [
        [
            f"{g['gpu']} @ {g['power']}",
            g["containers"],
            "" if g["share"] is None else f"{100 * g['share']:.0f}%",
            g["guard_misses"],
            f"{g['minutes']:.1f}",
        ]
        for g in stats["groups"]
    ]


def md_table(headers: list[str], rows: list[list]) -> str:
    def cell(x) -> str:
        return str("" if x is None else x).replace("|", "\\|").replace("\n", " ")

    out = ["| " + " | ".join(headers) + " |", "|" + " --- |" * len(headers)]
    out += ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def text_table(headers: list[str], rows: list[list]) -> str:
    cells = [[str("" if c is None else c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]
    numeric = [
        all(re.fullmatch(r"[+-]?\d+(\.\d+)?\*?", r[i]) for r in cells if r[i])
        for i in range(len(headers))
    ]

    def line(r: list[str]) -> str:
        parts = [
            c.rjust(w) if numeric[i] else c.ljust(w) for i, (c, w) in enumerate(zip(r, widths))
        ]
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(r) for r in cells]
    return "\n".join(out)


# ----------------------------------------------------------------------------- markdown
def render_markdown(ctx: dict) -> str:
    variants, controls, stats = ctx["variants"], ctx["controls"], ctx["ledger_stats"]
    gpu_line = ctx["gpu_used_line"]
    powers: dict[str, int] = {}
    for c in controls:
        powers[c["gpu_at_power"]] = powers.get(c["gpu_at_power"], 0) + 1
    power_note = ", ".join(f"{k} x{v}" for k, v in sorted(powers.items())) or "none"
    k_used_rows = sum(1 for v in variants if v["k_source"] == "k_used")
    lines = [
        "# CIFAR-100 speedrun leaderboard",
        "",
        f"- Generated {ctx['generated']} from `{ctx['registry']}`: "
        f"{len(variants)} variant rows, {len(controls)} control rows.",
        f"- k = {ctx['k_text']}",
        f"- {gpu_line} (ledger `{ctx['ledger']}`: {stats['containers']} containers, "
        f"{stats['guard_misses']} guard misses, {stats['guard_miss_minutes']:.1f} min lost).",
        "",
        "Score: `dtime_adj = dtime - dacc_pp / k` (seconds; dtime = variant minus control mean "
        "prepare+train time, dacc in accuracy percentage points, both from paired trials); lower "
        "is better. dtime_adj is recomputed from the current k for every row with paired data; "
        "rows without paired data sit at the bottom sorted by dtime. "
        f"`{FLAG}` marks nonfinite > 0 or a verdict that is not qualified/complete; "
        "`*` marks a dtime_adj computed from the row's own k_used (no k.json)"
        + (f" ({k_used_rows} rows)." if k_used_rows else "."),
        "",
        "## Variants (lower dtime_adj is better)",
        "",
        md_table(VARIANT_COLUMNS, [variant_cells(v, compact=False) for v in variants])
        if variants
        else "_No variant rows._",
        "",
        f"## Controls (power limits seen: {power_note})",
        "",
        md_table(CONTROL_COLUMNS, [control_cells(c) for c in controls])
        if controls
        else "_No control rows._",
        "",
        "## GPU hit rates (from the ledger)",
        "",
    ]
    if stats["containers"]:
        lines.append(
            md_table(
                ["GPU @ power", "containers", "share", "guard misses", "GPU min"],
                hit_rate_cells(stats),
            )
        )
        lines.append("")
        lines.append(
            f"Guard misses: {stats['guard_misses']}/{stats['containers']} containers "
            f"({100 * stats['guard_miss_rate']:.0f}%), {stats['guard_miss_minutes']:.1f} GPU min "
            f"lost; ledger total {stats['total_minutes']:.1f} min."
        )
    else:
        lines.append("_Ledger empty or missing._")
    lines.append("")
    lines.append("Pareto plot: `pareto.svg` (x = mean prepare+train s, y = mean accuracy %).")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- SVG
def nice_axis(lo: float, hi: float, target: int = 6) -> tuple[float, float, list[float], float]:
    """Round [lo, hi] out to tick multiples of a 1/2/2.5/5 step; returns (lo, hi, ticks, step)."""
    if not hi > lo:
        lo, hi = lo - 1.0, hi + 1.0
    raw = (hi - lo) / max(target - 1, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-12)
    start = math.floor(lo / step + 1e-9) * step
    end = math.ceil(hi / step - 1e-9) * step
    n = int(round((end - start) / step))
    ticks = [round(start + i * step, 10) for i in range(n + 1)]
    return ticks[0], ticks[-1], ticks, step


def fmt_tick(t: float, step: float) -> str:
    for d in range(0, 7):
        if abs(round(step, d) - step) < 1e-9:
            return f"{t:.{d}f}"
    return f"{t:g}"


def pareto_front(points: list[dict]) -> set[int]:
    """Indexes of points no other point beats on both time (lower) and accuracy (higher)."""

    def dominates(q: dict, p: dict) -> bool:
        return q["t"] <= p["t"] and q["a"] >= p["a"] and (q["t"] < p["t"] or q["a"] > p["a"])

    return {i for i, p in enumerate(points) if not any(dominates(q, p) for q in points)}


def esc(s) -> str:
    return escape(str(s), {'"': "&quot;"})


def boxes_overlap(a: tuple, b: tuple) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def place_labels(points: list[dict], width: int, height: int) -> list[tuple]:
    """Pick a label offset per point, alternating sides and dodging placed labels and markers."""
    char_w, line_h = 5.6, 10.0
    obstacles = [(p["px"] - 7, p["py"] - 7, p["px"] + 7, p["py"] + 7) for p in points]
    offsets = [(8, -5, "start"), (8, 13, "start"), (-8, -5, "end"), (-8, 13, "end")]
    placed: list[tuple] = []
    for i, p in enumerate(sorted(points, key=lambda q: (q["px"], q["py"]))):
        order = offsets if i % 2 == 0 else offsets[1:] + offsets[:1]
        w = char_w * len(p["label"])
        chosen = None
        for dx, dy, anchor in order:
            tx, ty = p["px"] + dx, p["py"] + dy
            x0 = tx if anchor == "start" else tx - w
            box = (x0, ty - line_h, x0 + w, ty + 2)
            inside = box[0] >= 2 and box[2] <= width - 2 and box[1] >= 2 and box[3] <= height - 2
            if inside and not any(boxes_overlap(box, b) for b in obstacles):
                chosen = (tx, ty, anchor, box)
                break
        if chosen is None:  # every slot collides: take the default and accept the overlap
            dx, dy, anchor = order[0]
            tx, ty = p["px"] + dx, p["py"] + dy
            x0 = tx if anchor == "start" else tx - w
            chosen = (tx, ty, anchor, (x0, ty - line_h, x0 + w, ty + 2))
        obstacles.append(chosen[3])
        placed.append((p, chosen[0], chosen[1], chosen[2]))
    return placed


def point_tooltip(p: dict, on_front: bool) -> str:
    e = p["entry"]
    kind = "control" if e["is_control"] else "variant"
    rnd = e["round"] or "?"
    tip = f"{e['label']} ({kind}, round {rnd}): {p['a']:.2f} % acc, {p['t']:.2f} s"
    if e["dacc_pp"] is not None:
        tip += f"; dacc {e['dacc_pp']:+.2f}"
        if e["dacc_se_pp"] is not None:
            tip += f" +- {e['dacc_se_pp']:.2f}"
        tip += " pp"
    if e["dtime"] is not None:
        tip += f"; dtime {e['dtime']:+.2f} s"
    if e["dtime_adj"] is not None:
        tip += f"; dtime_adj {e['dtime_adj']:+.2f} s"
    if e["verdict"]:
        tip += f"; {e['verdict']}"
    if on_front:
        tip += "; on the Pareto frontier"
    return tip


def render_svg(ctx: dict) -> str:
    width, height = 800, 500
    ml, mr, mb = 62, 24, 52
    entries = ctx["controls"] + ctx["variants"]
    points = [
        {"t": e["mean_time"], "a": e["mean_acc_pct"], "label": e["label"], "entry": e}
        for e in entries
        if e["mean_time"] is not None and e["mean_acc_pct"] is not None
    ]
    rounds = sorted({p["entry"]["round"] for p in points}, key=natural_key)
    colour = {r: (SERIES[i] if i < len(SERIES) else MUTED) for i, r in enumerate(rounds)}
    confirmed = [
        p["t"]
        for p in points
        if not p["entry"]["is_control"] and verdict_confirmed(p["entry"]["verdict"])
    ]
    best_confirmed = min(confirmed) if confirmed else None

    # Legend (wraps when many rounds); the plot top depends on the number of legend rows.
    items = [("circle", colour[r], f"round {r}" if r else "no round") for r in rounds]
    items += [("square", INK_2, "control"), ("line", INK_2, "Pareto frontier")]
    items += [("dash", MUTED, "75 % target")]
    if best_confirmed is not None:
        items.append(("dash", MUTED, "best CONFIRMED time"))
    legend, lx, ly = [], ml, 44
    for kind, col, text in items:
        w = 16 + 6.0 * len(text) + 18
        if lx + w > width - mr and lx > ml:
            lx, ly = ml, ly + 16
        legend.append((lx, ly, kind, col, text))
        lx += w
    mt = ly + 22
    pw, ph = width - ml - mr, height - mt - mb

    xs = [p["t"] for p in points] or [0.0, 10.0]
    ys = [p["a"] for p in points] + [75.0] if points else [70.0, 80.0]

    def padded(vals: list[float]) -> tuple[float, float]:
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or max(abs(hi) * 0.1, 1.0)
        return lo - span * 0.07, hi + span * 0.07

    x0, x1, xticks, xstep = nice_axis(*padded(xs))
    y0, y1, yticks, ystep = nice_axis(*padded(ys))

    def sx(t: float) -> float:
        return ml + (t - x0) / (x1 - x0) * pw

    def sy(a: float) -> float:
        return mt + ph - (a - y0) / (y1 - y0) * ph

    for p in points:
        p["px"], p["py"] = sx(p["t"]), sy(p["a"])
    front = pareto_front(points)

    text_attrs = f'font-family="{FONT}"'
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" {text_attrs} font-size="11" fill="{INK_2}">',
        "<title>CIFAR-100 speedrun: accuracy vs prepare+train time</title>",
        f'<rect width="{width}" height="{height}" fill="{SURFACE}"/>',
        f'<text x="{ml}" y="24" font-size="14" font-weight="600" fill="{INK}">'
        "CIFAR-100 speedrun: mean accuracy vs mean prepare+train time</text>",
        f'<text x="{width - mr}" y="24" text-anchor="end" font-size="10" fill="{MUTED}">'
        f"{esc(ctx['generated'])}; k = {esc(ctx['k_short'])}</text>",
    ]
    for lx, ly, kind, col, text in legend:
        if kind == "circle":
            out.append(f'<circle cx="{lx + 6}" cy="{ly}" r="5" fill="{col}"/>')
        elif kind == "square":
            out.append(f'<rect x="{lx + 1}" y="{ly - 5}" width="10" height="10" fill="{col}"/>')
        elif kind == "line":
            out.append(
                f'<line x1="{lx}" y1="{ly}" x2="{lx + 12}" y2="{ly}" stroke="{col}" stroke-width="2"/>'
            )
        else:
            out.append(
                f'<line x1="{lx}" y1="{ly}" x2="{lx + 12}" y2="{ly}" stroke="{col}" '
                'stroke-width="1.5" stroke-dasharray="4 3"/>'
            )
        out.append(f'<text x="{lx + 16}" y="{ly + 4}" fill="{INK_2}">{esc(text)}</text>')

    # Grid, axes, ticks.
    for t in xticks:
        x = sx(t)
        out.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{x:.1f}" y="{mt + ph + 16}" text-anchor="middle" fill="{MUTED}">'
            f"{fmt_tick(t, xstep)}</text>"
        )
    for a in yticks:
        y = sy(a)
        out.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(
            f'<text x="{ml - 8}" y="{y:.1f}" dy="0.35em" text-anchor="end" fill="{MUTED}">'
            f"{fmt_tick(a, ystep)}</text>"
        )
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt + ph}" stroke="{AXIS}"/>')
    out.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="{AXIS}"/>')
    out.append(
        f'<text x="{ml + pw / 2:.1f}" y="{height - 12}" text-anchor="middle" fill="{INK_2}">'
        "mean prepare + train time (s)</text>"
    )
    out.append(
        f'<text transform="translate(16 {mt + ph / 2:.1f}) rotate(-90)" text-anchor="middle" '
        f'fill="{INK_2}">mean accuracy (%)</text>'
    )

    # Reference lines: the 75 % target and the best CONFIRMED time.
    if y0 <= 75.0 <= y1:
        y = sy(75.0)
        out.append(
            f'<line x1="{ml}" y1="{y:.1f}" x2="{ml + pw}" y2="{y:.1f}" stroke="{MUTED}" '
            'stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{ml + pw - 4}" y="{y - 5:.1f}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}">75.0 % target</text>'
        )
    if best_confirmed is not None:
        x = sx(best_confirmed)
        label = f"best CONFIRMED {best_confirmed:.2f} s"
        anchor = "start" if x + 6 + 5.6 * len(label) < width - mr else "end"
        tx = x + 5 if anchor == "start" else x - 5
        out.append(
            f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt + ph}" stroke="{MUTED}" '
            'stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        out.append(
            f'<text x="{tx:.1f}" y="{mt + 12}" text-anchor="{anchor}" font-size="10" '
            f'fill="{MUTED}">{esc(label)}</text>'
        )

    # Pareto frontier, then marks (controls as squares, variants as circles), then labels.
    front_pts = sorted((points[i] for i in front), key=lambda p: (p["t"], -p["a"]))
    if len(front_pts) >= 2:
        pts = " ".join(f"{p['px']:.1f},{p['py']:.1f}" for p in front_pts)
        out.append(
            f'<polyline points="{pts}" fill="none" stroke="{INK_2}" stroke-width="1.5" '
            'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    if not points:
        out.append(
            f'<text x="{ml + pw / 2:.1f}" y="{mt + ph / 2:.1f}" text-anchor="middle" '
            f'fill="{MUTED}">no rows with mean time and accuracy</text>'
        )
    for i, p in enumerate(points):
        e, col = p["entry"], colour[p["entry"]["round"]]
        out.append(f"<g><title>{esc(point_tooltip(p, i in front))}</title>")
        out.append(
            f'<circle cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" r="12" fill="#000" fill-opacity="0"/>'
        )
        if e["is_control"]:
            out.append(
                f'<rect x="{p["px"] - 5:.1f}" y="{p["py"] - 5:.1f}" width="10" height="10" '
                f'fill="{col}" stroke="{SURFACE}" stroke-width="2"/>'
            )
        else:
            out.append(
                f'<circle cx="{p["px"]:.1f}" cy="{p["py"]:.1f}" r="5" fill="{col}" '
                f'stroke="{SURFACE}" stroke-width="2"/>'
            )
        out.append("</g>")
    for p, tx, ty, anchor in place_labels(points, width, height):
        out.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="{anchor}" font-size="10" '
            f'fill="{INK_2}">{esc(p["label"])}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- main
def json_row(e: dict) -> dict:
    d = dict(e["raw"])
    d.update({k: e[k] for k in COMPUTED_KEYS})
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--registry", type=Path, default=RUNS_DIR / "registry.jsonl")
    ap.add_argument("--ledger", type=Path, default=RUNS_DIR / "gpu_ledger.jsonl")
    ap.add_argument("--k-file", type=Path, default=RUNS_DIR / "k.json")
    ap.add_argument("--out-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--budget-min", type=float, default=DEFAULT_BUDGET_MIN)
    ap.add_argument("--json", action="store_true", help="print the sorted rows as JSON")
    ap.add_argument("--no-write", action="store_true", help="skip leaderboard.md and pareto.svg")
    args = ap.parse_args(argv)

    now = datetime.now().astimezone()
    k, k_data = load_k(args.k_file)
    previous = k_data.get("previous") if isinstance(k_data, dict) else None
    for block in previous if isinstance(previous, list) else [previous]:
        if isinstance(block, dict) and num(block.get("k")):
            ROUND_K.update({str(r): float(block["k"]) for r in block.get("rounds", [])})
    rows = read_jsonl(args.registry, "registry")
    ledger = read_jsonl(args.ledger, "ledger")
    entries = [build_entry(r, k) for r in rows]
    variants = sorted((e for e in entries if not e["is_control"]), key=sort_key)
    for i, e in enumerate(variants, 1):
        e["rank"] = i
    controls = sorted((e for e in entries if e["is_control"]), key=lambda e: (e["time"], e["job"]))
    if k is None and any(v["dtime_adj"] is None and v["dtime"] is not None for v in variants):
        warn("no k available for some paired rows: their dtime_adj cells are blank")
    stats = ledger_stats(ledger)
    gpu_used_line = f"GPU used: {stats['total_minutes']:.1f}/{args.budget_min:.0f} min"
    if k is not None:
        extra = ", ".join(str(k_data[f]) for f in ("source", "time") if k_data.get(f))
        k_text = f"{k:.4g} pp/s from `{rel(args.k_file)}`" + (f" ({extra})" if extra else "")
        k_short = f"{k:.3g} pp/s"
    else:
        k_text = "none (k.json missing or invalid; dtime_adj uses each row's k_used where present)"
        k_short = "n/a"
    ctx = {
        "generated": now.strftime("%Y-%m-%d %H:%M"),
        "registry": rel(args.registry),
        "ledger": rel(args.ledger),
        "k_text": k_text,
        "k_short": k_short,
        "variants": variants,
        "controls": controls,
        "ledger_stats": stats,
        "gpu_used_line": gpu_used_line,
    }
    outputs: dict[str, str | None] = {"leaderboard_md": None, "pareto_svg": None}
    if not args.no_write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        md_path, svg_path = args.out_dir / "leaderboard.md", args.out_dir / "pareto.svg"
        md_path.write_text(render_markdown(ctx), encoding="utf-8", newline="\n")
        svg_path.write_text(render_svg(ctx), encoding="utf-8", newline="\n")
        outputs = {"leaderboard_md": rel(md_path), "pareto_svg": rel(svg_path)}

    if args.json:
        payload = {
            "generated": now.isoformat(timespec="seconds"),
            "registry": rel(args.registry),
            "ledger": rel(args.ledger),
            "k_file": rel(args.k_file),
            "k": k,
            "k_info": k_data or None,
            "budget_min": args.budget_min,
            "gpu_minutes": stats["total_minutes"],
            "gpu_used_line": gpu_used_line,
            "rows": [json_row(v) for v in variants],
            "controls": [json_row(c) for c in controls],
            "gpu_hit_rates": stats,
            "outputs": outputs,
        }
        print(json.dumps(payload, indent=1))
        return 0

    if variants:
        print(text_table(VARIANT_COLUMNS, [variant_cells(v, compact=True) for v in variants]))
    else:
        print("no variant rows in the registry")
    print()
    print(f"k = {k_text.replace('`', '')}")
    print(
        f"{gpu_used_line} ({stats['containers']} containers, {stats['guard_misses']} guard "
        f"misses, {stats['guard_miss_minutes']:.1f} min lost)"
    )
    if outputs["leaderboard_md"]:
        print(f"wrote {outputs['leaderboard_md']} and {outputs['pareto_svg']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
