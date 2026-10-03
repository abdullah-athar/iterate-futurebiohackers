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

    # Without target_length, any length is valid, including the empty string
    free = ProblemInstance(name="free", strings=["ACGT", "AGCT"], alphabet="ACGT")
    for candidate in ["", "A", "ACGTA"]:
        valid, msg = free.validate_candidate(candidate)
        assert valid, msg
    print("Validation test passed!")


def test_variable_length_benchmarks():
    print("Testing variable-length scoring...")
    # Generated instances must not fix the consensus length
    for tier in ("small", "medium", "hard"):
        for inst in get_benchmark_suite(tier):
            assert inst.target_length is None, inst.name

    evaluator = Evaluator(default_tier="small")

    # Empty consensus is valid and costs the total read length
    summary = evaluator.evaluate_solver(lambda inst: "", benchmark="small", verbose=False)
    assert summary.all_valid
    assert summary.total_score == sum(len(s) for inst in get_benchmark_suite("small") for s in inst.strings)

    # "AAC" (not an input read) beats every read: cost 3 vs set median cost 4
    reads = ProblemInstance(name="aaa_acc_cac", strings=["AAA", "ACC", "CAC"], alphabet="ACGT")
    summary = evaluator.evaluate_solver(lambda inst: "AAC", benchmark=[reads], verbose=False)
    assert summary.all_valid
    assert summary.total_score == 3
    assert summary.total_baseline_score == 4

    # A shorter-than-planted consensus is accepted (previously rejected by target_length)
    inst = next(i for i in get_benchmark_suite("medium") if i.name == "dna_promoter_25bp")
    summary = evaluator.evaluate_solver(lambda i: i.strings[0][:20], benchmark=[inst], verbose=False)
    assert summary.all_valid
    print("Variable-length test passed!")


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


def test_public_view():
    print("Testing the public view handed to solvers...")
    inst = get_benchmark_suite("small")[1]
    view = inst.public_view()
    assert view.planted_consensus is None and view.known_best_score is None and view.metadata == {}
    assert view.description == "" and view.strings == inst.strings and view.strings is not inst.strings
    assert (view.num_strings, view.alphabet_set, view.metric) == (inst.num_strings, inst.alphabet_set, inst.metric)

    def mutate(i):
        i.strings[:] = i.strings[:1]
        return i.strings[0]

    evaluator = Evaluator()
    honest = evaluator.evaluate_solver(lambda i: i.strings[0], benchmark=[inst], verbose=False)
    mutated = evaluator.evaluate_solver(mutate, benchmark=[inst], verbose=False)
    assert honest.total_score > 0
    assert (mutated.total_score, mutated.total_baseline_score) == (honest.total_score, honest.total_baseline_score)
    assert inst.strings == get_benchmark_suite("small")[1].strings  # the original was never touched
    print("Public view test passed!")


if __name__ == "__main__":
    test_metrics()
    test_validation()
    test_variable_length_benchmarks()
    test_solvers_and_evaluator()
    test_public_view()
    print("\nAll tests passed successfully!")
