#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
OUT="${OUT:-docs/results/q1_leg_sidestep_6144_8000.mp4}"
bash play_native.sh --skill sidestep --headless --video "$OUT" "$@"
"$ROOT/.venv/bin/python" - "$OUT" <<'PY'
import json,sys,subprocess
import numpy as np
from pathlib import Path
video=Path(sys.argv[1])
metadata=json.loads(video.with_suffix('.json').read_text())
if not video.is_file() or video.stat().st_size<1024 or not metadata['survived'] or metadata['frames']<380:
    raise SystemExit('Recording did not complete a standing sidestep episode; inspect metadata before sharing.')
def frame(time):
    result=subprocess.run(['ffmpeg','-v','error','-ss',str(time),'-i',str(video),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'],check=True,capture_output=True)
    return np.frombuffer(result.stdout,dtype=np.uint8).reshape(720,1280,3).astype(float)
first=frame(.5)
change=max(np.mean(np.max(abs(first-frame(t)),axis=2)>20) for t in (2.5,4.5,6.5))
if change<.001:raise SystemExit('Video frames appear frozen; inspect viewport rendering before sharing.')
print(f"Recorded {metadata['duration']:.2f}s: {video}")
PY
if [[ "${UPLOAD_GOFILE:-0}" == 1 ]]; then
  "$ROOT/.venv/bin/python" "$ROOT/tools/upload_gofile.py" "$OUT"
fi
