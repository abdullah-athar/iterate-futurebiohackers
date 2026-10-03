#!/usr/bin/env bash
# Run a command inside the speedrun env from WSL (Windows hosts only; Linux/macOS use `uv run` directly).
#   scripts/wsl_speedrun.sh python -m benchmark.run --submission futurebiohackers --device cpu --synthetic --n 2
# The venv lives on the Linux filesystem ($HOME/.venvs/cifar100-speedrun): fast, not synced by
# cloud-sync tools, and it never collides with a Windows .venv in the same folder.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/cifar100-speedrun}"
cd "$(dirname "$(readlink -f "$0")")/../cifar100-speedrun"
exec uv run --frozen "$@"
