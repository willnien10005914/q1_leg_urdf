#!/usr/bin/env python3
"""Finish terminal status publication for an already-running supervisor."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
args=p.parse_args()
while True:
    try:
        state=json.loads((ROOT/'outputs/interactive_training_status.json').read_text())
    except (OSError,ValueError):
        time.sleep(15);continue
    if state.get('run')!=args.run:break
    if state.get('state') in ('complete','artifact_error'):break
    if state.get('state') in ('budget_exhausted','runtime_error','interrupted'):
        subprocess.run([sys.executable,str(ROOT/'tools/publish_interactive_status.py')],cwd=ROOT,check=True)
        break
    time.sleep(15)
