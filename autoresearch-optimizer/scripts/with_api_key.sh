#!/usr/bin/env bash
# Run any command (swarm, benchmark, calibration) on an Anthropic API key instead of the claude.ai login.
#
# Why a wrapper: with a claude.ai login present, `claude -p` keeps using the login and ignores
# ANTHROPIC_API_KEY (and with no login it ignores an unapproved key: apiKeySource "none"). An isolated
# config dir (no login) whose settings only say `apiKeyHelper: printenv AR_ANTHROPIC_KEY` makes every
# claude call of the run — agent sessions, describe and plan calls — authenticate with the key.
# The key itself is never written to disk: it lives in this process's environment only.
#
# The key must be scoped to a workspace (an organisation-level key needs an anthropic-workspace-id header
# on every request). Key source, first found: $AR_ANTHROPIC_KEY, then the macOS Keychain item named by
# $AR_KEYCHAIN_ITEM (default anthropic-api-key-workspace; store it yourself with
#   security add-generic-password -a "$USER" -s anthropic-api-key-workspace -w   # prompts, no echo
# ).
#   scripts/with_api_key.sh --check                       # one tiny call: prints the key source and cost
#   scripts/with_api_key.sh scripts/bench_policies.sh div1
set -eu
CONFIG=${AR_CLAUDE_CONFIG_DIR:-$HOME/.claude-apikey}
mkdir -p "$CONFIG"
printf '{"apiKeyHelper": "printenv AR_ANTHROPIC_KEY"}\n' > "$CONFIG/settings.json"
if [ -z "${AR_ANTHROPIC_KEY:-}" ]; then
  AR_ANTHROPIC_KEY=$(security find-generic-password -s "${AR_KEYCHAIN_ITEM:-anthropic-api-key-workspace}" -w 2>/dev/null || true)
fi
if [ -z "$AR_ANTHROPIC_KEY" ]; then
  echo "no key: set AR_ANTHROPIC_KEY or store it in the Keychain (see the header of this script)" >&2
  exit 2
fi
export AR_ANTHROPIC_KEY CLAUDE_CONFIG_DIR="$CONFIG"
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
if [ "${1:-}" = "--check" ]; then
  cd /tmp && claude -p --output-format stream-json --verbose --model haiku "Reply OK" 2>&1 | python3 -c '
import json, sys
for line in sys.stdin:
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("subtype") == "init":
        print("key source:", d.get("apiKeySource"), "(expected: apiKeyHelper)")
    if d.get("type") == "result":
        print("call:", "error" if d.get("is_error") else "ok", "| cost $", d.get("total_cost_usd"), "|", str(d.get("result"))[:80])'
  exit 0
fi
exec "$@"
