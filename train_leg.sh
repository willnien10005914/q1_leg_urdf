#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/q1-leg-mpl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
exec "${Q1_PYTHON:-$PWD/.venv/bin/python}" -u tools/train.py "$@"
