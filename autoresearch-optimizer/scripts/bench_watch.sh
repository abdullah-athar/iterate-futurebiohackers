#!/usr/bin/env bash
# Live view of a running benchmark: regenerate artifacts/bench/TAG.html every SECONDS (runs in progress
# included, the page reloads itself and keeps your show/hide choices), open it in Safari once, and stop
# after a last render when no `bench.sh TAG` process is left.
#   scripts/bench_watch.sh TAG [SECONDS]
set -u
cd "$(dirname "$0")/.."
TAG=$1 S=${2:-60}
opened=0
while :; do
  uv run python scripts/bench_report.py "$TAG" --html --live "$S" >/dev/null 2>&1 || true
  if [ $opened = 0 ] && [ -f "artifacts/bench/$TAG.html" ]; then open -a Safari "artifacts/bench/$TAG.html"; opened=1; fi
  pgrep -f "bench.sh $TAG" >/dev/null || break
  sleep "$S"
done
uv run python scripts/bench_report.py "$TAG" --html >/dev/null 2>&1  # final page, no auto-reload
echo "final report: artifacts/bench/$TAG.html"
