#!/usr/bin/env bash
# Model grid: Sonnet, Opus and Sopus (Sonnet for generation 1, then Opus) on each benchmark, repeated
# over swarm seeds. Runs one after the other (parallel runs would compete for CPU during evaluation)
# and skips any run that already finished, so it can be restarted.
#   scripts/model_grid.sh [TAG]      # runs land in artifacts/runs/<TAG>-<problem>-<config>-s<seed>
# Then: uv run python -m autoresearch_viz grid 'artifacts/runs/<TAG>-*' -o results/<TAG>.html
set -u
cd "$(dirname "$0")/.."
TAG=${1:-grid}
AGENTS=${AGENTS:-4}
GENERATIONS=${GENERATIONS:-3}
SEEDS=${SEEDS:-"0 1 2"}
PROBLEMS=${PROBLEMS:-"median_string median_string_long"}
CONFIGS=${CONFIGS:-"sonnet:sonnet opus:opus sopus:sonnet,opus"}

for seed in $SEEDS; do
  for problem in $PROBLEMS; do
    for cfg in $CONFIGS; do
      name=${cfg%%:*} model=${cfg#*:}
      dir=artifacts/runs/$TAG-$problem-$name-s$seed
      if grep -qs '"type": "run_end"' "$dir/events.jsonl"; then
        echo "skip $dir (finished)"
        continue
      fi
      echo "== $dir (model $model)"
      uv run python -m autoresearch --run "$dir" swarm --problem "$problem" --agents "$AGENTS" \
        --generations "$GENERATIONS" --eval local --max-budget-usd 1 --seed "$seed" --model "$model" 2>&1 \
        | grep -E "gen [0-9]+ end|holdout|run end|Error|error" | tail -6
    done
  done
done
