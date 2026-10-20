#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
export Q1_LEG_CURRICULUM_CONFIG="$ROOT/config/sidestep_foot_lift.json"
diagnostic_marker="$(mktemp /tmp/q1-foot-diagnostic.XXXXXX)"
trap 'rm -f "$diagnostic_marker"' EXIT
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/inspect_foot_lift.py --headless "$@"
# Some Kit builds mask Python exceptions with exit status zero.
test -s "$ROOT/outputs/foot_lift_diagnostic.json"
test "$ROOT/outputs/foot_lift_diagnostic.json" -nt "$diagnostic_marker"
