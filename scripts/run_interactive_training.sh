#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
restore_live=0
if systemctl --user is-active --quiet q1-leg-live; then
  restore_live=1
  systemctl --user stop q1-leg-live
fi
if [[ -f "$ROOT/outputs/interactive_live_restore.flag" ]]; then restore_live=1; fi
restore_viewer() {
  if [[ -f "$ROOT/tools/publish_interactive_status.py" ]]; then
    "$ROOT/.venv/bin/python" "$ROOT/tools/publish_interactive_status.py" || true
  fi
  if [[ "$restore_live" == 1 ]]; then bash "$ROOT/start_native_web.sh" || true; fi
  rm -f "$ROOT/outputs/interactive_live_restore.flag"
}
trap restore_viewer EXIT
"$ROOT/.venv/bin/python" -u "$ROOT/tools/run_interactive_training.py" "$@"
