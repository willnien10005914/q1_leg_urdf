#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-leg-training; then
  printf 'Q1 leg training is already running. Check: systemctl --user status q1-leg-training\n'
  exit 0
fi
mkdir -p "$ROOT/outputs"
exec systemd-run --user --unit=q1-leg-training --collect --description='Q1 native leg PPO curriculum' \
  --property=KillMode=control-group --property=TimeoutStopSec=30 --property="StandardOutput=append:$ROOT/outputs/native_curriculum.log" \
  --property=StandardError=inherit --working-directory="$ROOT" bash "$ROOT/scripts/run_native_curriculum.sh" \
  "${Q1_LEG_TRAIN_CONFIG:-$ROOT/config/sidestep_conservative.json}" "${Q1_LEG_ATTEMPTS:-2}"
