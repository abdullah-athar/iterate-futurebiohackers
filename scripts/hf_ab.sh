#!/usr/bin/env bash
# Paired A/B on one Hugging Face Jobs A100 80GB: control (defaults) plus each variant,
# run sequentially in one container with the same seeds. A variant's "_team" key runs another
# submission folder (stage it with EXTRA_TEAMS="a b").
#   scripts/hf_ab.sh 8 '[{"epochs": 8}, {"batch_size": 1536}]'
set -euo pipefail
N="$1"; VARIANTS="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEAM="${TEAM:-futurebiohackers}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cd "$ROOT/cifar100-speedrun"
tar --exclude=__pycache__ -cf - pyproject.toml uv.lock .python-version LICENSE README.md \
  benchmark "submissions/$TEAM" ${EXTRA_TEAMS:+$(printf "submissions/%s " $EXTRA_TEAMS)} | tar -xf - -C "$STAGE"
cp "$ROOT/scripts/hf_ab_driver.py" "$STAGE/"
cd "$STAGE"
hf jobs run --flavor "${FLAVOR:-a100-large}" --timeout "${TIMEOUT:-2h}" \
  -v .:/src -e OMP_NUM_THREADS=4 -e SEED="${SEED:-0}" -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy \
  ghcr.io/astral-sh/uv:python3.12-bookworm \
  bash -c "set -e; cp -r /src /app && cd /app && uv sync --frozen --no-dev -q \
&& nvidia-smi --query-gpu=name,power.limit --format=csv \
&& /opt/venv/bin/python -m benchmark.data --root data >/dev/null \
&& /opt/venv/bin/python hf_ab_driver.py $TEAM $N $(printf '%q' "$VARIANTS")"
