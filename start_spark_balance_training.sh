#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ ! -x .venv/bin/python ]]; then
  echo "Run bash install.sh to create the local CPU supervisor environment." >&2
  exit 1
fi
.venv/bin/python tools/run_spark_balance.py --dry-run > /dev/null
mkdir -p outputs
systemd-run --user --unit=q1-leg-spark-balance-15000 \
  --description="Q1 Spark balance PPO: 6144 envs, 15000 new iterations" \
  --property=WorkingDirectory="$ROOT" --property=KillMode=control-group \
  --property=TimeoutStopSec=45 --property=Restart=no \
  --property="StandardOutput=append:$ROOT/outputs/spark_balance_supervisor.log" \
  --property="StandardError=append:$ROOT/outputs/spark_balance_supervisor.log" \
  "$ROOT/.venv/bin/python" -u "$ROOT/tools/run_spark_balance.py" "$@"
