# Team workspace commands. Run `just` to list them.
# Override the team folder with TEAM=name just run.

set working-directory := "cifar100-speedrun"

team := env("TEAM", "futurebiohackers")
upstream := "https://github.com/AIDDA-Institute/CIFAR-100-speedrun"

default:
    @just --list

# Install the pinned harness environment
setup:
    uv sync --frozen

# Download CIFAR-100 into cifar100-speedrun/data
data:
    uv run python -m benchmark.data --root data

# CPU check of our submission on synthetic images (no GPU or dataset needed)
smoke:
    uv run python -m benchmark.run --submission {{team}} --device cpu --synthetic --n 2

# Train and score our submission on the GPU: just run 3 --params '{"epochs": 10}'
[positional-arguments]
run n="1" *args:
    shift; uv run python -m benchmark.run --submission {{team}} --n {{n}} "$@"

# Like run, but report results without the 75% accuracy target
[positional-arguments]
diag n="1" *args:
    shift; uv run python -m benchmark.run --submission {{team}} --n {{n}} --no-accuracy-target "$@"

# Like run, but on a Modal A100-80GB from your laptop: just modal 3 --params '{"epochs": 10}'
[positional-arguments]
modal n="1" *args:
    shift; TEAM={{team}} uv run --project .. modal run ../scripts/modal_speedrun.py --n {{n}} "$@"

# Show the most recent run summary
last:
    cat "$(ls -d results/{{team}}/*/ | tail -1)summary.json"

# Lint and run the harness tests
check:
    uv run ruff check .
    uv run pytest

# Pull organizer updates into cifar100-speedrun/
sync:
    cd "$(git rev-parse --show-toplevel)" && git subtree pull --prefix=cifar100-speedrun {{upstream}} main

# Install the autoresearch optimizer's Python 3.11 environment
[working-directory: justfile_directory() / "autoresearch-optimizer"]
autoresearch-setup:
    uv sync --frozen

# Verify that the autoresearch workspace uses its own Python 3.11 environment
[working-directory: justfile_directory() / "autoresearch-optimizer"]
autoresearch-smoke:
    uv run --frozen python -c 'import sys; from pathlib import Path; assert sys.version_info[:2] == (3, 11), sys.version; assert Path(sys.prefix).resolve() == (Path.cwd() / ".venv").resolve(), sys.prefix; print("Autoresearch optimizer ready: Python", sys.version.split()[0], "in", Path.cwd())'

# Evaluate median string solvers: just autoresearch-eval --compare all --tier small
[working-directory: justfile_directory() / "autoresearch-optimizer"]
[positional-arguments]
autoresearch-eval *args:
    uv run python scripts/evaluate_median_string.py "$@"

