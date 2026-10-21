#!/usr/bin/env bash
# Same launch convention as wheel; each unit belongs only to this leg project.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! systemctl --user is-active --quiet q1-leg-web; then
  systemd-run --user --unit=q1-leg-web --collect --description='Q1 native leg web viewer' \
    --property=Restart=on-failure --property=RestartSec=5 --working-directory="$ROOT" "$ROOT/start_web.sh"
fi
DESIRED_CONTROLLER="$("$ROOT/.venv/bin/python" - "$ROOT" <<'PY'
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('q1_leg_player_selection', root/'scripts/reinforcement_learning/rsl_rl/play_interactive_web.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
version, _, gate = module.selected_policy(root)
print('q1_leg_'+version if gate['ready'] else 'legacy')
PY
)"
PLAYER="$ROOT/play_native_web.sh"
if [[ "$DESIRED_CONTROLLER" != legacy ]]; then
  PLAYER="$ROOT/play_interactive_web.sh"
fi
if systemctl --user is-active --quiet q1-leg-live; then
  CURRENT_CONTROLLER="$("$ROOT/.venv/bin/python" - <<'PY'
import json
from urllib.request import urlopen
try:
    with urlopen('http://127.0.0.1:8877/state', timeout=1.) as response:
        state = json.load(response)
    print(state.get('controller_version', 'legacy'))
except (OSError, ValueError):
    print('unknown')
PY
)"
  # A validated new generation needs its matching environment. Restart only
  # across generations, never replace weights in an incompatible running env.
  if [[ "$CURRENT_CONTROLLER" != unknown && "$CURRENT_CONTROLLER" != "$DESIRED_CONTROLLER" ]]; then
    systemctl --user stop q1-leg-live
  fi
fi
if ! systemctl --user is-active --quiet q1-leg-live; then
  systemd-run --user --unit=q1-leg-live --collect --description='Q1 native leg live PhysX PPO player' \
    --property=KillMode=control-group --property=TimeoutStopSec=15 --property="StandardOutput=append:$ROOT/outputs/native_live.log" \
    --property=StandardError=inherit --working-directory="$ROOT" bash "$PLAYER"
fi
printf 'Q1 Leg: http://127.0.0.1:8876/web/\n'
