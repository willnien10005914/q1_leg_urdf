#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT/scripts/isaac_env.sh"
cd "$ROOT"
OUT="${OUT:-docs/results/q1_leg_interactive_v4.mp4}"
"$ISAAC_PYTHON" -u scripts/reinforcement_learning/rsl_rl/record_interactive_v4.py --headless --video "$OUT" "$@"
"$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np

video=Path(sys.argv[1])
report=json.loads(video.with_suffix('.json').read_text())
checks=(video.is_file(),video.stat().st_size>1024,report.get('controller_version')=='q1_leg_v4',
        report.get('confirmation_episodes',0)>=128,report.get('succeeded') is True,
        report.get('survived') is True,report.get('actual_failed') is False,
        report.get('completed') is True,report.get('final_mode')=='stand',
        report.get('final_sequence_active') is False,report.get('no_reset_count')==0,
        report.get('auto_resets_suppressed')==0,
        report.get('frames')==report.get('expected_frames'),
        abs(report.get('duration',0)-report.get('expected_duration',0))<.03,
        report.get('video_sha256')==hashlib.sha256(video.read_bytes()).hexdigest())
if not all(checks):raise SystemExit('Incomplete or failed v4 recording. Inspect candidate metadata; do not publish.')
modes={item['mode'] for item in report['mode_changes']}
if not {'sidestep_left','sidestep_right'}.issubset(modes):
    raise SystemExit('Both finite sidesteps must appear in the recording.')
def frame(time):
    result=subprocess.run(['ffmpeg','-v','error','-ss',str(time),'-i',str(video),
        '-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
    return np.frombuffer(result.stdout,dtype=np.uint8).reshape(720,1280,3).astype(float)
first=frame(.5)
change=max(np.mean(np.max(abs(first-frame(t)),axis=2)>20) for t in (2.,6.,10.5,12.))
if change<.001:raise SystemExit('Viewport recording appears frozen. Do not publish.')
print(f"Recorded verified native v4 direction control + left/right sidestep: {report['duration']:.2f}s, {video}")
PY
