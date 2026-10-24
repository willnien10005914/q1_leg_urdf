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
if (( $# == 0 )); then
  UPPER_RESUME_RUN="$("$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);profile=json.loads((root/'config/upper_body_v6.json').read_text())
if profile.get('control_fork'):
    status=root/'outputs/upper_body_training_status.json'
    if not status.exists():raise SystemExit('A budgeted fork needs its initialized --resume-run; refusing a new zero-counter run')
    state=json.loads(status.read_text());run=Path(state['run']).resolve()
    if state.get('profile')!=profile or not run.is_relative_to((root/'checkpoints/isaac/upper_body_v6/runs').resolve()):
        raise SystemExit('No matching fork run is available; specify its verified --resume-run')
    if state.get('ready') is True:print('COMPLETE')
    else:print(run)
PY
)"
  if [[ "$UPPER_RESUME_RUN" == COMPLETE ]]; then printf 'V6 has completed validation; no extra training started.\n'; exit 0; fi
  if [[ -n "$UPPER_RESUME_RUN" ]]; then set -- --resume-run "$UPPER_RESUME_RUN"; fi
fi
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
