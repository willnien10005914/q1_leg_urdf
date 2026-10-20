#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${Q1_PYTHON:-$PWD/.venv/bin/python}"
VIDEOS_ONLY=0
if [[ "${1:-}" == --videos-only ]]; then VIDEOS_ONLY=1; shift; fi
mkdir -p docs/reference
if [[ ! -s docs/reference/sidestep_0_7.mp4 ]]; then
  "$PYTHON" -m yt_dlp --js-runtimes node --remote-components ejs:github --no-playlist \
    --download-sections '*0-7' --force-keyframes-at-cuts -f 'bv*[height<=720]/b[height<=720]' \
    -o 'docs/reference/sidestep_0_7.%(ext)s' 'https://www.youtube.com/watch?v=KKhF516rSyo'
fi
if [[ "$VIDEOS_ONLY" == 1 ]]; then
  exec "$PYTHON" tools/render_reference_overlay.py
fi
MODEL="${POSE_MODEL:-$PWD/docs/reference/pose_landmarker_heavy.task}"
if [[ ! -f "$MODEL" ]]; then
  curl --fail --location --retry 2 -o "$MODEL" \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/q1-leg-mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON" tools/extract_sidestep.py --model "$MODEL" "$@"
