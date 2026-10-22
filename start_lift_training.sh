#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-leg-lift-training; then
  printf 'Q1 lift training is already active; inspect systemctl --user status q1-leg-lift-training.\n'
  exit 0
fi
if systemctl --user is-active --quiet q1-leg-training; then
  printf 'Existing Q1 training owns the GPU; stop or finish that experiment first.\n' >&2
  exit 1
fi
mkdir -p "$ROOT/outputs"
if ! systemctl --user is-active --quiet q1-leg-validation-watchdog; then
  systemd-run --user --unit=q1-leg-validation-watchdog --collect \
    --description='Q1 validation timeout guard without PPO restart' \
    --property=Restart=on-failure --property=RestartSec=10 \
    --property="StandardOutput=append:$ROOT/outputs/lift_validation_watchdog.log" \
    --property=StandardError=inherit --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/watch_lift_validation.py" --enforce
fi
if ! systemctl --user is-active --quiet q1-leg-initialization-watchdog; then
  systemd-run --user --unit=q1-leg-initialization-watchdog --collect \
    --description='Q1 pre-PPO initialization timeout guard' \
    --property=Restart=on-failure --property=RestartSec=10 \
    --property="StandardOutput=append:$ROOT/outputs/lift_initialization_watchdog.log" \
    --property=StandardError=inherit --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/watch_lift_initialization.py" --enforce
fi
exec systemd-run --user --unit=q1-leg-lift-training --collect \
  --description='Q1 lift v5 curriculum PPO: 10240 envs, max 15000 cumulative updates' \
  --property=KillMode=control-group --property=TimeoutStopSec=30 \
  --property="StandardOutput=append:$ROOT/outputs/lift_training.log" --property=StandardError=inherit \
  --working-directory="$ROOT" "$ROOT/.venv/bin/python" -u "$ROOT/tools/run_lift_training.py" "$@"
