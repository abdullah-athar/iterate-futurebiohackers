#!/usr/bin/env bash
# The diversity protocol: four configurations on ONE commit, same problem, instances, seeds, model, agents,
# per-session cap and the same run-level money cap (which includes describe/plan calls):
#   A  reference: the corrected framework, no diversity policy (families described for diagnostics only;
#      that spend is logged as describe:diagnostic and excluded from A's cap, so it cannot shrink A's search)
#   B  distance only:     --distance-policy soft --diverse-parents
#   C  entropy + grace:   --entropy-controller --family-grace            (C does not say which of the two helps)
#   D  B + C
# Optional (set them to 1): WITH_JOHANN=1 adds J = Johann's layer (tag bench-johann, its own cost accounting);
# WITH_SPLIT=1 adds C1 = grace only and C2 = controller only, to separate C's two parts.
# Every arm passes --no-descriptors except J (Johann's layer is its own arm, never mixed with the scheduler).
#   scripts/bench_policies.sh TAG                 # e.g. TAG=div1
#   DRY_RUN=1 scripts/bench_policies.sh TAG       # print the plan
# Then: uv run python scripts/bench_report.py TAG --html --target-gain <fixed beforehand>
set -eu
cd "$(dirname "$0")/.."
TAG=${1:-div}
RUN_USD=${MAX_RUN_USD:-3}
CAP="--max-run-usd,$RUN_USD"
REF="--no-descriptors,$CAP"
B="--distance-policy,soft,--diverse-parents"
C="--entropy-controller,--family-grace"
ARMS_DEFAULT="A=HEAD,$REF,--family-diagnostics B=HEAD,$REF,$B C=HEAD,$REF,$C D=HEAD,$REF,$B,$C"
[ "${WITH_SPLIT:-0}" = 1 ] && ARMS_DEFAULT="$ARMS_DEFAULT C1=HEAD,$REF,--family-grace C2=HEAD,$REF,--entropy-controller"
[ "${WITH_JOHANN:-0}" = 1 ] && ARMS_DEFAULT="$ARMS_DEFAULT J=bench-johann"
# generations high on purpose: the money cap (and BUDGET_MIN) ends each run, so arms compare at equal spend
ARMS=${ARMS:-$ARMS_DEFAULT} GENERATIONS=${GENERATIONS:-30} AGENTS=${AGENTS:-6} BUDGET_MIN=${BUDGET_MIN:-60} \
  exec scripts/bench.sh "$TAG"
