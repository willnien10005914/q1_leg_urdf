#!/usr/bin/env bash
# Same launch convention as wheel; each unit belongs only to this leg project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! systemctl --user is-active --quiet q1-leg-web; then
  systemd-run --user --unit=q1-leg-web --collect --description='Q1 native leg web viewer' \
    --property=Restart=on-failure --property=RestartSec=5 --working-directory="$ROOT" "$ROOT/start_web.sh"
fi
if ! systemctl --user is-active --quiet q1-leg-live; then
  PLAYER="$ROOT/play_native_web.sh"
  if [[ -f "$ROOT/scripts/reinforcement_learning/rsl_rl/play_interactive_web.py" ]] && \
      "$ROOT/.venv/bin/python" "$ROOT/scripts/reinforcement_learning/rsl_rl/play_interactive_web.py" --validate-only >/dev/null 2>&1; then
    PLAYER="$ROOT/play_interactive_web.sh"
  fi
  systemd-run --user --unit=q1-leg-live --collect --description='Q1 native leg live PhysX PPO player' \
    --property=KillMode=control-group --property=TimeoutStopSec=15 --property="StandardOutput=append:$ROOT/outputs/native_live.log" \
    --property=StandardError=inherit --working-directory="$ROOT" bash "$PLAYER"
fi
printf 'Q1 Leg: http://127.0.0.1:8876/web/\n'
