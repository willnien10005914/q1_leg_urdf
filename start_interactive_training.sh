#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-leg-training; then
  printf 'Q1 leg training is already active; check systemctl --user status q1-leg-training.\n'
  exit 0
fi
mkdir -p "$ROOT/outputs"
exec systemd-run --user --unit=q1-leg-training --collect --description='Q1 v4 command PPO: 10240 envs, max 15000 iterations' \
  --property=KillMode=control-group --property=TimeoutStopSec=30 \
  --property="StandardOutput=append:$ROOT/outputs/interactive_training.log" --property=StandardError=inherit \
  --working-directory="$ROOT" bash "$ROOT/scripts/run_interactive_training.sh" "$@"
