#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/'; exit 1; }
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.10 .venv; fi
# MuJoCo verification uses CPU Torch. Isaac uses its own wheel-compatible runtime.
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r pyproject.toml --extra motion --extra test
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/q1-leg-mpl}"
.venv/bin/python tools/build_assets.py
echo 'Ready. Run bash prepare_motion.sh, then bash start_web.sh.'
