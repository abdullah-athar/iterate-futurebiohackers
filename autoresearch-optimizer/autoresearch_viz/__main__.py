"""CLI: `python -m autoresearch_viz render RUN... -o out.html` / `serve RUN...` / `grid RUN... -o grid.html`."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .html import render, render_main
from .load import load_runs
from .metrics import summarize


def _render(specs: list[str], out: Path, title: str | None, open_browser: bool) -> int:
    runs = load_runs(specs)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(runs, title) if title else render(runs))
    for s in (summarize(r) for r in runs):
        print(
            f"{s.label:45s} best={s.best_objective:>8.0f}  Δbaseline={s.gain_vs_baseline_pct:+6.1f}%  "
            f"evals={s.evaluated:3d}  dupes={s.counts.get('rejected_duplicate', 0):2d}  tokens={s.tokens:>8,d}"
        )
    print(f"wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _grid(specs: list[str], out: Path, open_browser: bool) -> int:
    from .grid import config_name, render_grid

    runs = load_runs(specs)
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_grid(runs))
    for r in runs:
        s = summarize(r)
        print(f"{r.path.name:45s} {r.problem:20s} {config_name(r):8s} best={s.best_objective:>8.0f} "
              f"gain={s.gain_vs_seed_pct:+5.1f}% cost=${s.cost_usd:.2f}")
    print(f"wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return 0


def _serve(specs: list[str], port: int, title: str | None, open_browser: bool) -> int:
    """Serve a live dashboard: the page re-fetches /fragment every 3 s; re-rendered only when run files change."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    title = title or "Autoresearch — live run"
    cache: dict[str, tuple[tuple, bytes]] = {}

    def stamp() -> tuple:
        files = [f for spec in specs for f in Path(spec.rpartition("=")[2] or spec).glob("**/*.json*")]
        return tuple(sorted((str(f), f.stat().st_mtime_ns) for f in files if f.name in ("ledger.jsonl", "events.jsonl", "holdout.json")))

    def page(kind: str) -> bytes:
        key = stamp()
        if kind not in cache or cache[kind][0] != key:
            runs = load_runs(specs)
            body = render(runs, title, live=True) if kind == "page" else render_main(runs, title)
            cache[kind] = (key, body.encode())
        return cache[kind][1]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            kind = "fragment" if self.path.startswith("/fragment") else "page" if self.path in ("/", "/index.html") else None
            if kind is None:
                self.send_error(404)
                return
            try:
                body = page(kind)
            except FileNotFoundError as e:
                body = f"<p>waiting for run data: {e}</p>".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"live dashboard at {url} (Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="autoresearch_viz", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render one HTML dashboard comparing the given run directories")
    r.add_argument("runs", nargs="+", help="run dir (contains ledger.jsonl), a parent dir of runs, or label=path")
    r.add_argument("-o", "--out", type=Path, default=Path("artifacts/viz/dashboard.html"))
    r.add_argument("--title")
    r.add_argument("--open", action="store_true", help="open the result in a browser")
    v = sub.add_parser("serve", help="serve a live dashboard that refreshes while a run is in progress")
    v.add_argument("runs", nargs="+", help="run dir(s), a parent dir of runs, or label=path")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--title")
    v.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    g = sub.add_parser("grid", help="compare configurations (e.g. Sonnet / Opus / Sopus) across benchmarks and repeats")
    g.add_argument("runs", nargs="+", help="run dirs or a parent dir of runs")
    g.add_argument("-o", "--out", type=Path, default=Path("artifacts/viz/grid.html"))
    g.add_argument("--open", action="store_true", help="open the result in a browser")
    a = p.parse_args(argv)
    if a.cmd == "grid":
        return _grid(a.runs, a.out, a.open)
    if a.cmd == "serve":
        return _serve(a.runs, a.port, a.title, a.open)
    return _render(a.runs, a.out, a.title, a.open)


if __name__ == "__main__":
    sys.exit(main())
