#!/usr/bin/env python3
"""Required physical-hardware cohort, separate from ordinary Cargo CI.

This Unix host-owner runner opens an EXISTING shared lease, verifies contention,
then selects every required ignored case by exact identity. No success receipt is
possible from skipped, filtered, absent, timed-out or unsuccessful cases.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

CASES = (
    ('integration', 'native_gpu_general_only_control'),
    ('unit', 'gpu_group_update::tests::gpu_group_update_known_moments'),
    ('integration', 'native_gpu_bifactor_mixed_control'),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', type=Path, required=True)
    parser.add_argument('--integration', type=Path, required=True)
    parser.add_argument('--unit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {'started': datetime.datetime.now().astimezone().isoformat(),
               'required': list(CASES), 'cases': [], 'accepted': False}
    env = dict(os.environ)
    env.pop('FMLS_DS_GPU_LEASED', None)
    try:
        bins = {'integration': args.integration.resolve(strict=True),
                'unit': args.unit.resolve(strict=True)}
        receipt['binaries'] = {k: {'path': str(p), 'sha256': digest(p)} for k,p in bins.items()}
        # Discovery cannot dispatch work and must establish explicit enrollment.
        for key, binary in bins.items():
            out = subprocess.run([str(binary), '--list', '--ignored'], env=env,
                                 capture_output=True, text=True, timeout=30, check=True)
            (args.output / (key + '.listing')).write_text(out.stdout)
            registered = {line.removesuffix(': test') for line in out.stdout.splitlines()
                          if line.endswith(': test')}
            required = {name for target,name in CASES if target == key}
            if not required <= registered:
                raise RuntimeError('required hardware cases are not explicitly registered')
        # Never create, replace or unlink a shared lease file. The host owner
        # supplies the same path used by other GPU producers on this machine.
        with args.lease.open('r+') as lease:
            if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
                raise RuntimeError('lease is not a regular file')
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt['lease'] = {'path': str(args.lease.resolve()),
                                'inode': os.fstat(lease.fileno()).st_ino}
            contender = subprocess.run([sys.executable, '-c',
                'import fcntl,sys; f=open(sys.argv[1],"r+");\n'
                'try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'
                'except BlockingIOError: sys.exit(73)', str(args.lease)],
                env=env, timeout=30, check=False)
            receipt['contender_exit'] = contender.returncode
            if contender.returncode != 73:
                raise RuntimeError('independent lease contender did not refuse')
            env['FMLS_DS_GPU_LEASED'] = '1'
            for index, (key, name) in enumerate(CASES):
                binary = bins[key]
                if digest(binary) != receipt['binaries'][key]['sha256']:
                    raise RuntimeError('test binary changed after admission')
                command = [str(binary), '--ignored', '--exact', name, '--test-threads=1']
                row = {'target': key, 'case': name, 'argv': command, 'deadline_seconds': 240}
                start = time.monotonic()
                stdout = args.output / f'{index}.stdout'
                stderr = args.output / f'{index}.stderr'
                with stdout.open('w') as out, stderr.open('w') as err:
                    child = subprocess.Popen(command, env=env, stdout=out, stderr=err,
                        start_new_session=True, pass_fds=(lease.fileno(),))
                    row['pid'] = child.pid
                    try:
                        row['exit'] = child.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        row['timeout'] = True
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        row['exit'] = child.returncode
                row['elapsed'] = time.monotonic()-start
                row['child_settled'] = child.poll() is not None
                row['streams'] = {'stdout': digest(stdout), 'stderr': digest(stderr)}
                text = stdout.read_text()
                row['accepted'] = (row['exit'] == 0 and not row.get('timeout', False)
                    and f'test {name} ... ok' in text
                    and '1 passed; 0 failed; 0 ignored;' in text)
                receipt['cases'].append(row)
                if not row['accepted']:
                    raise RuntimeError('required hardware case did not pass')
            receipt['accepted'] = len(receipt['cases']) == len(CASES)
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        receipt['error_type'] = type(error).__name__
        receipt['error'] = str(error)
    finally:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'accepted': receipt['accepted'], 'executed': len(receipt['cases']),
                      'receipt': str(args.output / 'receipt.json')}))
    return 0 if receipt['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main())
