#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
exec "$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/play_native.py "$@"
