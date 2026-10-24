#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
for arg in "$@"; do
  if [[ "$arg" == --validate-only ]]; then
    exec "$ROOT/.venv/bin/python" scripts/reinforcement_learning/rsl_rl/record_upper_body_v6.py "$@"
  fi
done
# Reject unavailable policies using the CPU interpreter before loading Kit.
"$ROOT/.venv/bin/python" scripts/reinforcement_learning/rsl_rl/record_upper_body_v6.py --validate-only
source "$ROOT/scripts/isaac_env.sh"
OUT="${OUT:-docs/results/q1_leg_upper_body_v6.mp4}"
Q1_UPPER_RECORDING_RUN_ID="$("$ROOT/.venv/bin/python" -c 'import uuid; print(uuid.uuid4().hex)')"
export Q1_UPPER_RECORDING_RUN_ID
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/record_upper_body_v6.py --headless --video "$OUT" "$@"
"$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path('scripts/reinforcement_learning/rsl_rl').resolve()))
from record_upper_body_v6 import ROOT,verify_artifact
video=Path(sys.argv[1])
report=verify_artifact(video,root=ROOT,run_id=os.environ['Q1_UPPER_RECORDING_RUN_ID'])
print(f"Verified native v6 walking + finite left/right + stand: {report['actual_duration_s']:.2f}s, {video}")
print('Recorded unloaded sole peaks L/R (m):',report['foot_metrics']['unloaded_peak_clearance_m_per_foot'])
PY
