# Median String / Steiner String Optimization Benchmark

A benchmark harness and evaluation framework for automated algorithm discovery and heuristic search on the **Median String problem** (also known as the **Steiner String problem** or **Generalized Median String problem**).

---

## The Problem

Given a set of target strings $S = \{s_1, s_2, \dots, s_k\}$ over an alphabet $\Sigma$ (e.g. DNA $\Sigma = \{A, C, G, T\}$ or Amino Acids), find a candidate string $t \in \Sigma^*$ that **minimizes the sum of distances** to all sequences in $S$:

$$\min_{t} \sum_{i=1}^k d(t, s_i)$$

Where $d(t, s_i)$ is typically the **Levenshtein (edit) distance** (insertions, deletions, substitutions) or **Hamming distance**.

### Why This Is Great for Autoresearch
1. **NP-Hard & Rich Landscape**: Unlike toy functions, finding the optimal Steiner string is NP-hard. Simple heuristics often get trapped in local optima.
2. **Clear Objective**: Minimize the total distance. Lower is strictly better.
3. **Rigorous Baseline**: The **Set Median** (choosing the best string from $S$ itself) guarantees a 2-approximation in metric spaces. Any true generalized median algorithm must beat the set median!
4. **Fast Iteration**: Evaluation of an instance takes milliseconds, allowing an AI optimizer to run dozens of experiments in minutes.
5. **Real Biological Impact**: Used in DNA regulatory motif finding, multiple sequence alignment consensus, phylogenetic ancestral sequence reconstruction, and error correction in high-throughput sequencing.

---

## Directory Structure

```text
autoresearch-optimizer/median_string/
├── README.md               # You are here
├── __init__.py             # Public API exports
├── __main__.py             # CLI entrypoint (`python -m median_string`)
├── instance.py             # ProblemInstance definition and constraint validation
├── metrics.py              # Levenshtein/Hamming distance, set median baseline, scoring
├── benchmarks.py           # Synthetic planted motifs and benchmark suites (small/medium/hard)
├── base_solver.py          # BaseSolver abstract class & FunctionalSolver wrapper
├── evaluator.py            # Evaluation harness, leaderboard comparison, JSON reporter
└── solvers/                # Folder for community and AI-generated solvers
    ├── __init__.py         # Solver registry & discovery
    ├── set_median.py       # Set median 2-approximation baseline
    ├── frequency_consensus.py  # Positional majority-vote consensus
    ├── random_solver.py    # Naive random search baseline
    └── template_solver.py  # Starter template with 1-edit local search
```

---

## Quickstart

Run commands from `autoresearch-optimizer/` using `uv run`:

### 1. Evaluate the baseline
```bash
uv run python -m median_string --solver set_median --tier small
```

### 2. Compare all registered solvers on a leaderboard
```bash
uv run python scripts/evaluate_median_string.py --compare all --tier small
```

Output:
```text
=====================================================================================
SOLVER LEADERBOARD COMPARISON (Tier: small)
=====================================================================================
Rank | Solver                   | Score   | vs Baseline  | Record (W/T/L) | Time (s) | Valid
--------------------------------------------------------------------------------------------
1    | template                 | 51      | +7 (+12.1%)  | 2/1/0          | 0.0529   | 3/3
2    | set_median               | 58      | +0 (+0.0%)   | 0/3/0          | 0.0027   | 3/3
3    | frequency_consensus      | 71      | -13 (-22.4%) | 1/1/1          | 0.0001   | 3/3
4    | random_baseline          | 137     | -79 (-136.2%) | 0/0/3          | 0.0238   | 3/3
=====================================================================================
```

### 3. Run on the standard benchmark tier (`medium`)
```bash
uv run python scripts/evaluate_median_string.py --compare set_median,template --tier medium
```

### 4. Save results to JSON for experiment tracking
```bash
uv run python scripts/evaluate_median_string.py --compare all --tier small --output artifacts/run_01.json
```

---

## How to Add Your Own Solution

Adding a new algorithm takes 3 simple steps:

### Method A: Add a solver in `solvers/`

1. Create a new file, e.g. `median_string/solvers/my_genetic_solver.py`:

```python
from median_string.base_solver import BaseSolver
from median_string.instance import ProblemInstance
from median_string.solvers import register_solver
from median_string.metrics import sum_distance

@register_solver("my_genetic_solver")
class MyGeneticSolver(BaseSolver):
    name = "my_genetic_solver"
    description = "Genetic algorithm for Steiner string search."

    def solve(self, instance: ProblemInstance) -> str:
        # instance.strings: list of sequences S
        # instance.alphabet: permitted chars (e.g. "ACGT")
        # instance.target_length: None for generated benchmarks (any length allowed)
        # instance.metric: "levenshtein" or "hamming"
        
        # Implement your search logic here:
        best_candidate = instance.strings[0]
        return best_candidate
```

2. Import your solver in `median_string/solvers/__init__.py`:
```python
from .my_genetic_solver import MyGeneticSolver
```

3. Immediately benchmark it from the command line:
```bash
uv run python -m median_string --solver my_genetic_solver --tier small
```

---

### Method B: Use a Python script or notebook directly

You can pass any function `fn(instance: ProblemInstance) -> str` directly to `Evaluator`:

```python
from median_string import Evaluator, ProblemInstance

def my_heuristic(instance: ProblemInstance) -> str:
    # Your algorithm here
    return instance.strings[0]

evaluator = Evaluator(default_tier="small")
summary = evaluator.evaluate_solver(my_heuristic, benchmark="small")
print(f"Total score: {summary.total_score}, vs baseline: {summary.net_improvement}")
```

---

## Benchmark Tiers

| Tier | Instances | Sequence Lengths | Number of Strings | Typical Runtime | Purpose |
| --- | --- | --- | --- | --- | --- |
| `small` | 3 | 10 – 16 bp/aa | 5 – 8 | < 0.1s | Fast smoke testing & inner search loops |
| `medium` | 5 | 20 – 50 bp/aa | 10 – 15 | ~ 1s – 2s | Standard validation benchmark |
| `hard` | 5 | 35 – 80 bp/aa | 15 – 40 | ~ 5s – 10s | Stress-testing scalability and heavy noise |

---

## Evaluation Criteria & Rules

Every proposed candidate string must satisfy:
1. **Alphabet compliance**: All characters in candidate must belong to `instance.alphabet`.
2. **Any length**: The candidate may be shorter or longer than the input strings, or empty. Generated benchmarks don't set `instance.target_length`; a custom instance can set it to require `len(candidate) == instance.target_length`.
3. **Primary Metric**: Total Steiner distance $\sum_{s \in S} d(\text{candidate}, s)$ across all instances (lower is better).
4. **Secondary Metrics**: Improvement % over the Set Median baseline, win rate (W/T/L), and execution runtime.
