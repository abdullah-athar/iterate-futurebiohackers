#!/usr/bin/env bash
# Run the CIFAR-100 speedrun harness on a Hugging Face Jobs A100 80GB (a100-large).
#   HF_TOKEN=... scripts/hf_run.sh --n 3 --no-accuracy-target --params '{"epochs":12}'
# Arguments are passed to `benchmark.run`. Defaults to our submission folder.
# Override hardware with FLAVOR=l40sx1 etc. Requires an HF account with Jobs access.
set -euo pipefail
SPEEDRUN="$(cd "$(dirname "$0")/../cifar100-speedrun" && pwd)"
TEAM="${TEAM:-futurebiohackers}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cd "$SPEEDRUN"
tar --exclude=__pycache__ -cf - pyproject.toml uv.lock .python-version LICENSE README.md benchmark \
  "submissions/$TEAM" | tar -xf - -C "$STAGE"
cd "$STAGE"
args=$(printf '%q ' "$@")
hf jobs run --flavor "${FLAVOR:-a100-large}" --timeout "${TIMEOUT:-2h}" \
  -v .:/src -e OMP_NUM_THREADS=4 -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "set -e; cp -r /src /app && cd /app && uv sync --frozen --no-dev -q \
&& nvidia-smi --query-gpu=name,memory.total --format=csv \
&& /opt/venv/bin/python -m benchmark.data --root data >/dev/null \
&& /opt/venv/bin/python -m benchmark.run --submission $TEAM $args \
&& cat results/$TEAM/*/summary.json"
