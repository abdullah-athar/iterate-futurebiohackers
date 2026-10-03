"""Tests for the autoresearch loop (offline: uses the mock proposer)."""

import sys
import tempfile
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from autoresearch.core import Candidate, SuiteSpec
from autoresearch.evaluators import SubprocessEvaluator
from autoresearch.loop import IterationBudget, run_loop
from autoresearch.memory import JsonlLedger
from autoresearch.proposers import MockProposer, extract_code, seed_candidate
from autoresearch.selection import GreedySelector
from autoresearch.suites import HELDOUT_SEED_OFFSET, build_suite

SMALL = SuiteSpec("small", 0)


def test_heldout_suite_differs():
    print("Testing held-out suite...")
    search, heldout = build_suite(SMALL), build_suite(SuiteSpec("small", HELDOUT_SEED_OFFSET))
    assert len(search) == len(heldout)
    for a, b in zip(search, heldout):
        assert a.strings != b.strings
        assert len(a.strings) == len(b.strings) and a.alphabet == b.alphabet
    print("  Held-out suite OK")


def test_evaluator_failures():
    print("Testing evaluator failure handling...")
    with tempfile.TemporaryDirectory() as tmp:
        ev = SubprocessEvaluator(Path(tmp), timeout_s=5)
        assert ev.evaluate(seed_candidate(), SMALL).valid

        syntax = ev.evaluate(Candidate(code="def broken(:\n"), SMALL)
        assert not syntax.valid and "SyntaxError" in syntax.error

        no_class = ev.evaluate(Candidate(code="x = 1\n"), SMALL)
        assert not no_class.valid and "No BaseSolver subclass" in no_class.error

        hang = (
            "from median_string.base_solver import BaseSolver\n"
            "class Hang(BaseSolver):\n"
            "    def solve(self, instance):\n"
            "        while True: pass\n"
        )
        timeout = SubprocessEvaluator(Path(tmp), timeout_s=1).evaluate(Candidate(code=hang), SMALL)
        assert not timeout.valid and "Timeout" in timeout.error
    print("  Evaluator failures OK")


def test_extract_code():
    assert extract_code("why\n```python\nprint(1)\n```") == "print(1)\n"
    assert extract_code("no code") is None


def test_mock_loop():
    print("Testing mock loop end to end...")
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        memory = JsonlLedger(run_dir)
        report = run_loop(
            seed=seed_candidate(),
            proposer=MockProposer(seed=0),
            evaluator=SubprocessEvaluator(run_dir, timeout_s=30),
            selector=GreedySelector(),
            memory=memory,
            budget=IterationBudget(3),
            search=SMALL,
            heldout=SuiteSpec("small", HELDOUT_SEED_OFFSET),
        )
        assert len(memory.records) == 4
        assert len((run_dir / "ledger.jsonl").read_text().splitlines()) == 4
        assert (run_dir / "champion.py").exists()
        assert report.heldout.valid
        assert report.champion.result.score <= memory.records[0].result.score
    print("  Mock loop OK")


if __name__ == "__main__":
    test_heldout_suite_differs()
    test_evaluator_failures()
    test_extract_code()
    test_mock_loop()
    print("\nAll autoresearch tests passed.")
