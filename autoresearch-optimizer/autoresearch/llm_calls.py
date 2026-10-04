"""Structured `claude -p` calls made outside the agent sessions (describe, plan), with a usage log.

Every provider call appends one JSON line to `<run>/llm_usage.jsonl`:
  kind (e.g. "describe", "plan", "describe:diagnostic"), model, ok, seconds, attempt,
  cost_usd (None when the provider did not report it: unknown, never silently 0),
  usage = the provider's token categories as reported (input, output, cache read, cache creation).
Output tokens already include any reasoning, so nothing is added on top. A local cache hit makes no
provider call and is not logged here (the caller counts cache hits separately).

Calls run from the temp dir: from inside the repo, claude also loads the project CLAUDE.md
(1374 vs 615 input tokens on a minimal call), which costs money and leaks unrelated instructions.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

_LOG_LOCK = threading.Lock()
TOKEN_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def log_usage(path: Path | None, rec: dict[str, Any]) -> None:
    if path is None:
        return
    with _LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def read_usage(root: Path) -> list[dict[str, Any]]:
    """All auxiliary-call records of a run (llm_usage.jsonl, plus the older describe_usage.jsonl)."""
    out = []
    for name in ("llm_usage.jsonl", "describe_usage.jsonl"):
        p = root / name
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    rec.setdefault("kind", "describe")
                    out.append(rec)
    return out


def claude_json(prompt: str, schema: dict, system: str, model: str, *, kind: str,
                usage_path: Path | None = None, timeout: float = 180, attempt: int = 1,
                budget=None) -> tuple[dict | None, dict]:
    """One structured call. Returns (structured_output or None, usage record). `budget` (a RunBudget)
    is checked before the call and charged after it; a refused call returns (None, rec with ok=False)."""
    rec: dict[str, Any] = {"t": time.time(), "kind": kind, "model": model, "attempt": attempt, "ok": False,
                           "seconds": 0.0, "cost_usd": None, "usage": {}}
    ticket = None
    if budget is not None:
        ticket = budget.reserve_aux(kind)
        if ticket is None:
            rec["error"] = "budget exhausted"
            return None, rec
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", "--output-format", "json", "--model", model, "--tools", "",
             "--system-prompt", system, "--json-schema", json.dumps(schema)],
            input=prompt, capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
        stdout, stderr = proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        stdout, stderr = "", "timeout"
    rec["seconds"] = round(time.time() - t0, 2)
    out = None
    try:
        res = json.loads(stdout)
        cost = res.get("total_cost_usd")
        rec["cost_usd"] = None if cost is None else float(cost)
        rec["usage"] = {k: v for k, v in (res.get("usage") or {}).items() if k in TOKEN_FIELDS}
        out = res.get("structured_output")
        rec["ok"] = isinstance(out, dict)
        if not rec["ok"]:
            rec["error"] = (res.get("subtype") or "no structured_output")[:200]
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError) as e:
        rec["error"] = f"unparseable output ({e}): {stderr[-200:]}"
    log_usage(usage_path, rec)
    if budget is not None:
        budget.settle_aux(ticket, rec)
    return out, rec
