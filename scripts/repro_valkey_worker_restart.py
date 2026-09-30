"""Bound and timestamp each phase of the owned disposable Valkey probe."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

parser = argparse.ArgumentParser()
parser.add_argument('phase', choices=('before', 'after'))
parser.add_argument('--port', required=True, type=int)
parser.add_argument('--state', required=True)
parser.add_argument('--evidence', required=True, type=Path)
args = parser.parse_args()
env = dict(os.environ)
env.pop('PYTHONOPTIMIZE', None)
env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
started = datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
clock = time.monotonic()
command = [sys.executable, str(Path(__file__).with_name('valkey_worker_restart_probe.py')),
           args.phase, '--port', str(args.port), '--state', args.state]
try:
    result = subprocess.run(command, env=env, timeout=60, capture_output=True, text=True)
    code, output, error = result.returncode, result.stdout, result.stderr
except subprocess.TimeoutExpired:
    code, output, error = 124, '', 'phase exceeded 60-second process deadline'
record = {'phase': args.phase, 'started_kst': started,
          'finished_kst': datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
          'elapsed_seconds': round(time.monotonic()-clock, 3), 'exit_code': code,
          'stdout': output.strip(), 'stderr': error.strip(),
          'scope': 'local editable worker, production MC dispatch, remote disposable daemon'}
with args.evidence.open('x') as file:
    json.dump(record, file, indent=2)
print(json.dumps(record))
sys.exit(code)
