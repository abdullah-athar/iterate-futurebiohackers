"""Run-level caps on money, provider calls and tokens, shared by agent sessions and auxiliary calls.

Spend is recomputed from what the run has written: ledger entries (one per agent session, including
sessions that produced nothing) and llm_usage.jsonl (describe/plan calls). A cost the provider did not
report is counted as unknown, not as zero. Work in flight is covered by reservations: a generation
reserves its agents' estimated cost before launching, and every auxiliary call reserves its own
estimate, so parallel work cannot overshoot a cap. Wall-clock stays with the swarm's --budget-min.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from .ledger import RunStore
from .llm_calls import read_usage

DEFAULT_AUX_USD = 0.05  # per describe/plan call before any has been measured in this run


@dataclass
class Spend:
    usd: float = 0.0
    unknown_cost: int = 0          # calls whose cost was not reported
    calls: int = 0                 # provider calls: agent sessions + auxiliary calls
    tokens: int = 0
    by_kind: dict[str, float] = field(default_factory=dict)   # "agent", "describe", "plan", ...


def run_spend(store: RunStore) -> Spend:
    s = Spend()
    for e in store.entries():
        if e.proposer == "seed" or not e.usage:   # the seed is not an agent session
            continue
        s.calls += 1
        s.tokens += e.prompt_tokens + e.completion_tokens
        cost = e.usage.get("cost_usd")
        if cost is None:
            s.unknown_cost += 1
        else:
            s.usd += float(cost)
            s.by_kind["agent"] = s.by_kind.get("agent", 0.0) + float(cost)
    for rec in read_usage(store.root):
        s.calls += int(rec.get("calls", 1))
        s.tokens += sum(int(v or 0) for v in (rec.get("usage") or {}).values())
        cost = rec.get("cost_usd")
        if cost is None:
            s.unknown_cost += 1
        else:
            s.usd += float(cost)
            kind = rec.get("kind", "describe").split(":")[0]
            s.by_kind[kind] = s.by_kind.get(kind, 0.0) + float(cost)
    return s


class RunBudget:
    def __init__(self, store: RunStore, max_usd: float | None = None, max_calls: int | None = None,
                 max_tokens: int | None = None) -> None:
        self.store, self.max_usd, self.max_calls, self.max_tokens = store, max_usd, max_calls, max_tokens
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._reserved_calls = 0
        self.refusals: list[str] = []

    @property
    def capped(self) -> bool:
        return any(v is not None for v in (self.max_usd, self.max_calls, self.max_tokens))

    def spent(self) -> Spend:
        return run_spend(self.store)

    def _fits(self, s: Spend, usd: float, calls: int) -> str | None:
        """None if `usd`/`calls` more (on top of spend and reservations) stay within every cap."""
        if self.max_usd is not None and s.usd + self._reserved_usd + usd > self.max_usd + 1e-9:
            return f"cost cap ${self.max_usd:g} (spent ${s.usd:.2f}, in flight ${self._reserved_usd:.2f}, next ${usd:.2f})"
        if self.max_calls is not None and s.calls + self._reserved_calls + calls > self.max_calls:
            return f"call cap {self.max_calls} (made {s.calls}, in flight {self._reserved_calls}, next {calls})"
        if self.max_tokens is not None and s.tokens >= self.max_tokens:
            return f"token cap {self.max_tokens} (used {s.tokens})"
        return None

    def aux_estimate(self, kind: str) -> float:
        costs = [r["cost_usd"] for r in read_usage(self.store.root)
                 if r.get("cost_usd") is not None and r.get("kind", "").split(":")[0] == kind.split(":")[0]]
        return max(costs) if costs else DEFAULT_AUX_USD

    # ----- a whole generation of agent sessions ---------------------------------------------
    def reserve_generation(self, n_agents: int, per_agent_usd: float | None) -> tuple[bool, str]:
        """Reserve n agent sessions. `per_agent_usd` None = no estimate yet (only the call cap applies)."""
        with self._lock:
            s = self.spent()
            usd = 0.0 if per_agent_usd is None or math.isnan(per_agent_usd) else n_agents * per_agent_usd
            why = self._fits(s, usd, n_agents)
            if why:
                self.refusals.append(why)
                return False, why
            self._reserved_usd += usd
            self._reserved_calls += n_agents
            return True, ""

    def release_generation(self) -> None:
        """The generation's sessions are now in the ledger: their spend is counted, drop the reservation."""
        with self._lock:
            self._reserved_usd = 0.0
            self._reserved_calls = 0

    # ----- auxiliary calls (llm_calls.claude_json) -----------------------------------------
    def reserve_aux(self, kind: str):
        if not self.capped:
            return (0.0, 0)
        est = self.aux_estimate(kind)
        with self._lock:
            why = self._fits(self.spent(), est, 1)
            if why:
                self.refusals.append(f"{kind}: {why}")
                return None
            self._reserved_usd += est
            self._reserved_calls += 1
            return (est, 1)

    def settle_aux(self, ticket, rec: dict) -> None:
        if not ticket or ticket == (0.0, 0):
            return
        with self._lock:  # the call is now in llm_usage.jsonl, so spent() includes it
            self._reserved_usd = max(0.0, self._reserved_usd - ticket[0])
            self._reserved_calls = max(0, self._reserved_calls - ticket[1])
