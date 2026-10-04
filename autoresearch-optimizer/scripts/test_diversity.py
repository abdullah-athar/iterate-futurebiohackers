"""Tests for the diversity extension (families, distances, scheduler) and the reference fixes it rests on.

The LLM is faked at the subprocess boundary (llm_calls.subprocess.run) or by passing descriptors directly;
evaluation is either the real local evaluator or a deterministic fake. Run: uv run python scripts/test_diversity.py
"""

from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import sys
import tempfile
import types
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from autoresearch.json_cache import JsonCache  # noqa: E402

DESCRIPTOR_OUT = {"terms": [{"term": "hill climbing", "tier": 6}], "new_terms": [], "summary": "s"}


class FakeClaude:
    """Stands in for subprocess.run inside llm_calls: answers every call with `answer(prompt)`."""

    def __init__(self, answer=lambda prompt: DESCRIPTOR_OUT, cost: float | None = 0.01):
        self.answer, self.cost, self.prompts = answer, cost, []
        self.TimeoutExpired = TimeoutError

    def run(self, cmd, input, **kw):
        assert kw.get("cwd") == tempfile.gettempdir(), kw.get("cwd")
        self.prompts.append(input)
        out = {"structured_output": self.answer(input), "usage": {"input_tokens": 100, "output_tokens": 20}}
        if self.cost is not None:
            out["total_cost_usd"] = self.cost
        return types.SimpleNamespace(stdout=json.dumps(out), stderr="")


class patched_claude:  # noqa: N801 - used as a context manager
    def __init__(self, fake: FakeClaude):
        self.fake = fake

    def __enter__(self):
        import autoresearch.llm_calls as L
        self.L, self.real = L, L.subprocess
        L.subprocess = self.fake
        return self.fake

    def __exit__(self, *exc):
        self.L.subprocess = self.real


def tmpdir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"autoresearch-{prefix}-"))


# ----- step 1: reference fixes -------------------------------------------------------------------

def test_describe_miss_then_hit():
    import autoresearch.descriptors as D
    tmp = tmpdir("describe")
    try:
        cache = tmp / "descriptors.json"
        with patched_claude(FakeClaude()) as fake:
            a = D.describe("def solve(i):\n    return 'a'\n", D.Vocabulary(), "p", cache_path=cache)
            b = D.describe("def solve(i):\n    return 'a'\n", D.Vocabulary(), "p", cache_path=cache)
        assert a.core == b.core == "hill climbing" and len(fake.prompts) == 1  # miss, then hit
        usage = [json.loads(l) for l in (tmp / "llm_usage.jsonl").read_text().splitlines()]
        assert len(usage) == 1 and usage[0]["kind"] == "describe" and usage[0]["cost_usd"] == 0.01
        assert usage[0]["usage"] == {"input_tokens": 100, "output_tokens": 20}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  describe: miss then hit, no deadlock, spend logged")


def test_describe_cache_invalidation():
    import autoresearch.descriptors as D
    src = "def solve(i):\n    return 'a'\n"
    keys = {D.describe_key(src, "p", "sonnet"), D.describe_key(src, "p", "haiku"), D.describe_key(src, "q", "sonnet")}
    assert len(keys) == 3, keys  # model and problem text are part of the key
    renamed = "def solve(inst):\n    # a comment\n    return 'a'\n"
    assert D.describe_key(src, "p", "sonnet") == D.describe_key(renamed, "p", "sonnet")  # cosmetic edits hit
    print("  describe: cache key changes with model/problem, not with cosmetic edits")


def test_unknown_cost_is_not_zero():
    import autoresearch.descriptors as D
    from autoresearch.budget import run_spend
    from autoresearch.ledger import RunStore
    tmp = tmpdir("unknown-cost")
    try:
        store = RunStore(tmp / "run")
        store.create({})
        with patched_claude(FakeClaude(cost=None)):
            D.describe("def solve(i):\n    return 'b'\n", D.Vocabulary(), "p", cache_path=store.root / "descriptors.json")
        rec = json.loads((store.root / "llm_usage.jsonl").read_text())
        assert rec["cost_usd"] is None
        s = run_spend(store)
        assert s.unknown_cost == 1 and s.usd == 0.0 and s.calls == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  usage: an unreported cost is logged as unknown (None), not 0")


def _writer(path: str, worker: int, n: int) -> None:
    cache = JsonCache(Path(path))
    for i in range(n):
        cache.put(f"w{worker}-{i}", {"worker": worker, "i": i})


def test_cache_concurrent_processes_and_threads():
    from concurrent.futures import ThreadPoolExecutor
    tmp = tmpdir("cache")
    try:
        path = tmp / "cache.json"
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=_writer, args=(str(path), w, 25)) for w in range(3)]
        for p in procs:
            p.start()
        with ThreadPoolExecutor(4) as pool:  # threads of this process write at the same time
            list(pool.map(lambda w: _writer(str(path), w, 25), range(3, 7)))
        for p in procs:
            p.join(60)
            assert p.exitcode == 0, p.exitcode
        data = json.loads(path.read_text())  # valid JSON: no partial write
        assert len(data) == 7 * 25, len(data)  # no lost update
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  cache: 3 processes + 4 threads, 175 writes, none lost, file always valid")


def test_budget_caps_with_calls_in_flight():
    from autoresearch.budget import RunBudget
    from autoresearch.ledger import Entry, RunStore
    tmp = tmpdir("budget")
    try:
        store = RunStore(tmp / "run")
        store.create({})
        store.append(Entry(id=0, parent_ids=[], mode="seed", hypothesis="", status="seed", proposer="seed"))
        for i in (1, 2):
            store.append(Entry(id=i, parent_ids=[0], mode="tune", hypothesis="", status="evaluated",
                               proposer="claude-code:x", generation=1, usage={"cost_usd": 0.40}))
        store.append(Entry(id=3, parent_ids=[0], mode="tune", hypothesis="", status="no_output",
                           proposer="claude-code:x", generation=1, usage={"cost_usd": 0.30}))
        b = RunBudget(store, max_usd=2.0)
        assert abs(b.spent().usd - 1.10) < 1e-9 and b.spent().calls == 3  # the empty session is counted
        ok, why = b.reserve_generation(2, 0.40)   # 1.10 + 0.80 <= 2.0
        assert ok, why
        assert b.reserve_aux("describe") is not None  # 1.10 + 0.80 in flight + 0.05 estimate
        assert b.reserve_aux("describe") is not None  # 2.00 exactly
        assert b.reserve_aux("describe") is None      # would exceed: refused while the others are in flight
        b.release_generation()
        ok, why = b.reserve_generation(3, 0.40)   # 1.10 + 1.20 > 2.0
        assert not ok and "cost cap" in why, why
        assert RunBudget(store, max_calls=3).reserve_generation(1, None)[0] is False  # call cap
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("  budget: caps hold with generation and auxiliary calls in flight")


TESTS = [test_describe_miss_then_hit, test_describe_cache_invalidation, test_unknown_cost_is_not_zero,
         test_cache_concurrent_processes_and_threads, test_budget_caps_with_calls_in_flight]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nAll {len(TESTS)} diversity tests passed!")
