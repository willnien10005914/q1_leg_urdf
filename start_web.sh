#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec "${Q1_PYTHON:-$PWD/.venv/bin/python}" web/serve.py "$@"
