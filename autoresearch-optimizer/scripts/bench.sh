#!/usr/bin/env bash
# Benchmark arms on the same problems, model and seeds:
#   initial = the framework before the exploration-exploitation layer (main at Johann's merge base)
#   johann  = Johann's layer with only the cache-deadlock fix and describe-cost logging (tag bench-johann)
#   mine    = HEAD of this branch (commit first: uncommitted edits are not benchmarked)
# Each arm runs from a frozen checkout of its commit (one git worktree per commit under $BENCH_CODE), so
# later edits never leak into an earlier arm; refs are resolved once, at start. Runs go one after the
# other (local evaluation competes for CPU), arms interleaved within each seed, and finished runs are
# skipped, so the script can be restarted.
#   scripts/bench.sh [TAG]        # runs land in artifacts/runs/<TAG>-<problem>-<arm>-s<seed>
#   DRY_RUN=1 scripts/bench.sh    # print the plan only
#   ARMS="johann=bench-johann mine=HEAD" SEEDS="3 4" scripts/bench.sh   # arm = name=git-ref[,extra flags]
# Then: uv run python scripts/bench_report.py <TAG>
set -eu
cd "$(dirname "$0")/.."
OPT=$PWD
REPO=$(git rev-parse --show-toplevel)
TAG=${1:-bench}
MODEL=${MODEL:-sonnet}
AGENTS=${AGENTS:-8}
GENERATIONS=${GENERATIONS:-3}
SEEDS=${SEEDS:-"0 1 2"}
PROBLEMS=${PROBLEMS:-"median_string"}
BUDGET_MIN=${BUDGET_MIN:-45}   # wall-clock cap per run, generous so the describe step never cuts a generation
MAX_USD=${MAX_USD:-1}          # per agent session
ARMS=${ARMS:-"initial=c4438c4 johann=bench-johann mine=HEAD"}
BENCH_CODE=${BENCH_CODE:-$(dirname "$REPO")/iterate-futurebiohackers-bench}
DRY_RUN=${DRY_RUN:-0}

if [[ " $ARMS " == *"=HEAD"* ]] && [ -n "$(git status --porcelain -- .)" ]; then
  echo "warning: uncommitted changes in autoresearch-optimizer are NOT part of arm 'mine' (it runs HEAD)"
fi

# resolve every arm once to name=sha[,flags] and freeze its code in a worktree
RESOLVED=""
for arm in $ARMS; do
  name=${arm%%=*} spec=${arm#*=}
  ref=${spec%%,*} flags=""
  [[ $spec == *,* ]] && flags=${spec#*,}
  sha=$(git rev-parse --short=10 "$ref^{commit}")
  code=$BENCH_CODE/$name-$sha
  echo "arm $name = $ref ($sha) ${flags//,/ } -> $code"
  RESOLVED="$RESOLVED $name=$sha${flags:+,$flags}"
  if [ "$DRY_RUN" != 1 ] && [ ! -d "$code" ]; then
    git worktree add --detach "$code" "$sha" >/dev/null
    (cd "$code/autoresearch-optimizer" && uv sync -q)
  fi
done

mkdir -p artifacts/bench-logs
for seed in $SEEDS; do
  for problem in $PROBLEMS; do
    for arm in $RESOLVED; do
      name=${arm%%=*} spec=${arm#*=}
      sha=${spec%%,*} flags=""
      [[ $spec == *,* ]] && flags=${spec#*,} && flags=${flags//,/ }
      dir=$OPT/artifacts/runs/$TAG-$problem-$name-s$seed
      if grep -qs '"type": "run_end"' "$dir/events.jsonl"; then
        echo "skip $dir (finished)"
        continue
      fi
      echo "== $dir ($sha, $MODEL, $AGENTS agents x $GENERATIONS generations)"
      [ "$DRY_RUN" = 1 ] && continue
      # an unfinished run would resume and add generations: keep it aside and start over
      [ -e "$dir" ] && mv "$dir" "$dir.unfinished-$(date +%H%M%S)"
      log=$OPT/artifacts/bench-logs/$TAG-$problem-$name-s$seed.log
      # shellcheck disable=SC2086
      (cd "$BENCH_CODE/$name-$sha/autoresearch-optimizer" && PYTHONUNBUFFERED=1 uv run python -m autoresearch \
        --run "$dir" swarm --problem "$problem" --agents "$AGENTS" --generations "$GENERATIONS" \
        --budget-min "$BUDGET_MIN" --eval local --max-budget-usd "$MAX_USD" --seed "$seed" --model "$MODEL" \
        $flags) >"$log" 2>&1 || true
      grep -E "gen [0-9]+ end|descriptors:|holdout|run end|Error|error" "$log" | tail -8 || true
      if grep -qs '"type": "run_end"' "$dir/events.jsonl"; then
        printf '{"tag": "%s", "arm": "%s", "sha": "%s", "flags": "%s", "problem": "%s", "seed": %s, "model": "%s", "agents": %s, "generations": %s}\n' \
          "$TAG" "$name" "$sha" "$flags" "$problem" "$seed" "$MODEL" "$AGENTS" "$GENERATIONS" >"$dir/bench.json"
      else
        echo "   run did not finish, see $log"
      fi
    done
  done
done
echo "report: uv run python scripts/bench_report.py $TAG"
