#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
OUT="${OUT:-outputs/interactive_diagnostic.mp4}"
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/record_interactive_diagnostic.py --headless --video "$OUT" "$@"
"$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import hashlib, importlib.util, json, sys
from pathlib import Path
video=Path(sys.argv[1])
sys.path.insert(0,str(Path('scripts/reinforcement_learning/rsl_rl').resolve()))
from record_interactive_diagnostic import validate_metadata
report=json.loads(video.with_suffix('.json').read_text())
if not validate_metadata(report) or not video.is_file() or video.stat().st_size<1024:
    raise SystemExit('Incomplete diagnostic capture; inspect local metadata before sharing.')
if report.get('video_sha256')!=hashlib.sha256(video.read_bytes()).hexdigest():
    raise SystemExit('Diagnostic video hash does not match its metadata.')
print(f"UNACCEPTED native diagnostic: {video}, {report['actual_duration_s']:.2f}s, failure={report['failure']}; robot={report.get('robot', 'q1_leg_interactive_v4')}, task={report.get('task', 'legacy_diagnostic')}; not a successful skill demonstration.")
if 'foot_metrics' in report:
    print('Measured unloaded sole peaks (left/right m):',report['foot_metrics']['unloaded_peak_clearance_m_per_foot'])
    print('Measured planned high-swing success fractions:',report['foot_metrics']['planned_high_swing_success_rate_per_foot'])
PY
