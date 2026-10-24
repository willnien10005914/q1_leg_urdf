#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
publish=()
if [[ "${1:-}" == --publish ]]; then
  publish=(--publish)
  shift
fi
if (( $# )); then printf 'Usage: bash start_upper_body_monitor.sh [--publish]\n' >&2; exit 2; fi
mkdir -p "$ROOT/outputs"
if ! systemctl --user is-active --quiet q1-leg-upper-progress-monitor; then
  systemd-run --user --unit=q1-leg-upper-progress-monitor --collect \
    --description='Q1 V6 monitored progress and bounded SSH publication' \
    --property=Restart=on-failure --property=RestartSec=10 \
    --property="StandardOutput=append:$ROOT/outputs/upper_body_progress_monitor.log" \
    --property=StandardError=inherit --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" -u tools/monitor_upper_body_v6.py --watch "${publish[@]}"
fi
if ! systemctl --user is-active --quiet q1-leg-upper-finalizer; then
  systemd-run --user --unit=q1-leg-upper-finalizer --collect \
    --description='Q1 V6 confirmed native video GIF and GoFile pipeline' \
    --property="StandardOutput=append:$ROOT/outputs/upper_body_finalizer_service.log" \
    --property=StandardError=inherit --working-directory="$ROOT" \
    "$ROOT/.venv/bin/python" -u tools/finalize_upper_body_v6.py --wait "${publish[@]}"
fi
