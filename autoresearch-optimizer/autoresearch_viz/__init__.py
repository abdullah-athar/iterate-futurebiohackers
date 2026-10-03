"""autoresearch_viz: render one or more autoresearch runs (ledger.jsonl) as a self-contained HTML dashboard.

The dashboard compares research "flavours" (runs with different loop settings or proposers)
on the same problem: objective vs. iterations / LLM tokens / wall-clock, per-instance gains,
outcome mix, a scoreboard, and the per-run research trajectory (hypothesis -> outcome).
"""

from .load import Entry, Run, load_runs

__all__ = ["Entry", "Run", "load_runs"]
