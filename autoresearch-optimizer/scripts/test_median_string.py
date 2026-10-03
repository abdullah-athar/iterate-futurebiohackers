"""Unit and integration tests for the Median String evaluation harness."""

import sys
from pathlib import Path

# Add project root to sys.path so median_string is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from median_string import (
    Evaluator,
    ProblemInstance,
    calculate_distance,
    compute_set_median,
    get_benchmark_suite,
    get_solver,
    hamming_distance,
    levenshtein_distance,
    list_solvers,
    sum_distance,
)


def test_metrics():
    print("Testing metrics...")
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "abc") == 0
    assert levenshtein_distance("ACGT", "ACGT") == 0
    assert levenshtein_distance("ACGT", "ACT") == 1
    assert hamming_distance("ACGT", "ACCT") == 1
    assert hamming_distance("ACGT", "AC") == 2

    strings = ["ACGT", "ACCT", "ACGA"]
    # Candidate "ACGT": dist to "ACGT" is 0, to "ACCT" is 1, to "ACGA" is 1 -> total 2
    assert sum_distance("ACGT", strings, metric="levenshtein") == 2

    # Set median of strings
    best_str, best_dist = compute_set_median(strings, metric="levenshtein")
    assert best_str in strings
    assert best_dist == 2
    print("Metrics test passed!")


def test_validation():
    print("Testing instance validation...")
    inst = ProblemInstance(name="test", strings=["ACGT", "AGCT"], alphabet="ACGT", target_length=4)
    valid, msg = inst.validate_candidate("ACGT")
    assert valid, msg

    valid, msg = inst.validate_candidate("ACGX")  # 'X' not in ACGT
    assert not valid
    assert "not in alphabet" in msg

    valid, msg = inst.validate_candidate("ACG")  # Length 3 instead of 4
    assert not valid
    assert "does not match target length" in msg
    print("Validation test passed!")


def test_solvers_and_evaluator():
    print("Testing solvers and evaluator...")
    solvers = list_solvers()
    print(f"Registered solvers: {solvers}")
    assert "set_median" in solvers
    assert "frequency_consensus" in solvers
    assert "random_baseline" in solvers
    assert "template" in solvers

    evaluator = Evaluator(default_tier="small")
    for s_name in solvers:
        summary = evaluator.evaluate_solver(s_name, benchmark="small", verbose=False)
        assert summary.all_valid, f"Solver {s_name} produced invalid solutions: {summary.num_valid}/{summary.num_instances}"
        assert summary.total_score > 0
        print(f"Solver '{s_name}' evaluated successfully. Total score: {summary.total_score}, time: {summary.total_time_seconds:.4f}s")

    # Test compare
    leaderboard = evaluator.compare_solvers(solvers, benchmark="small")
    assert len(leaderboard) == len(solvers)
    print("Solvers and evaluator test passed!")


if __name__ == "__main__":
    test_metrics()
    test_validation()
    test_solvers_and_evaluator()
    print("\nAll tests passed successfully!")
