#!/usr/bin/env bash
# Release the live viewer GPU allocation for training, then restore it on exit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
restore_live=0
if systemctl --user is-active --quiet q1-leg-live; then
  restore_live=1
  systemctl --user stop q1-leg-live
fi
restore_viewer() {
  if [[ "$restore_live" == 1 ]]; then
    bash "$ROOT/start_native_web.sh" || true
  fi
}
trap restore_viewer EXIT
"$ROOT/.venv/bin/python" -u "$ROOT/tools/native_curriculum.py" --config "$1" --attempts "$2"
