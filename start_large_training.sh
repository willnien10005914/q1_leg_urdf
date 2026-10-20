#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export Q1_LEG_TRAIN_CONFIG="$ROOT/config/stand_sidestep_6144_8000.json"
export Q1_LEG_ATTEMPTS=1
exec bash "$ROOT/start_native_training.sh"
