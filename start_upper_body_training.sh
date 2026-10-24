#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if systemctl --user is-active --quiet q1-leg-upper-training; then
  printf 'Q1 upper-body training is already active; inspect systemctl --user status q1-leg-upper-training.\n'
  exit 0
fi
for unit in q1-leg-training q1-leg-lift-training q1-leg-live; do
  if systemctl --user is-active --quiet "$unit"; then
    printf 'Existing %s owns native GPU execution; finish or stop that process before starting V6.\n' "$unit" >&2
    exit 1
  fi
done
mkdir -p "$ROOT/outputs"
systemd-run --user --unit=q1-leg-upper-training --collect \
  --description='Q1 upper-body v6 native PPO: 10240 envs, max 20000 NEW updates' \
  --property=KillMode=control-group --property=TimeoutStopSec=30 \
  --property="StandardOutput=append:$ROOT/outputs/upper_body_training.log" --property=StandardError=inherit \
  --working-directory="$ROOT" "$ROOT/.venv/bin/python" -u "$ROOT/tools/run_upper_body_training.py" "$@"
if ! systemctl --user is-active --quiet q1-leg-upper-startup-watchdog; then
  systemd-run --user --unit=q1-leg-upper-startup-watchdog --collect \
    --description='Q1 V6 pre-PPO startup-only bounded recovery' \
    --property=Restart=on-failure --property=RestartSec=10 \
    --property="StandardOutput=append:$ROOT/outputs/upper_body_startup_watchdog.log" \
    --property=StandardError=inherit --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" -u "$ROOT/tools/watch_upper_body_startup.py" --enforce
fi
