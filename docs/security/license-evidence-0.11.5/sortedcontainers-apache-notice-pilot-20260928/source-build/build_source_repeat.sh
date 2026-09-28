#!/usr/bin/env bash
set -euo pipefail

inputs=$(cd "$(dirname "$0")/.." && pwd)
out=${1:?usage: build_source_repeat.sh NEW_OUTPUT_DIR}
if [ -e "$out" ]; then
  echo "Output already exists: $out" >&2
  exit 1
fi
mkdir -p "$out"
out=$(cd "$out" && pwd)

python3.12 - "$inputs" "$out" <<'PY'
import hashlib
import pathlib
import sys
import tarfile

inputs, out = map(pathlib.Path, sys.argv[1:])
expected = {
    'sortedcontainers-2.4.0.tar.gz': '25caa5a06cc30b6b83d11423433f65d1f9d76c4c6a0c90e3379eaa43b9bfdb88',
    'sortedcontainers-2.4.0-py2.py3-none-any.whl': 'a163dcaede0f1c021485e957a39245190e74249897e2ae4b2aa38595db237ee0',
    'LICENSE-2.0.txt': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30',
}
for name, digest in expected.items():
    assert hashlib.sha256((inputs / name).read_bytes()).hexdigest() == digest, name
with tarfile.open(inputs / 'sortedcontainers-2.4.0.tar.gz') as archive:
    members = archive.getmembers()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        assert not path.is_absolute() and '..' not in path.parts, member.name
        assert member.isfile() or member.isdir(), member.name
    for label in 'ABC':
        dest = out / f'source-{label}'
        dest.mkdir()
        archive.extractall(dest, filter='data')
        if label != 'A':
            (dest / 'sortedcontainers-2.4.0/LICENSE-APACHE-2.0.txt').write_bytes(
                (inputs / 'LICENSE-2.0.txt').read_bytes()
            )
PY

uv venv --python python3.12 "$out/.build-venv" > "$out/venv.log" 2>&1
uv pip install --python "$out/.build-venv/bin/python" --require-hashes --no-deps -r "$inputs/source-build/build-requirements.txt" > "$out/backend.log" 2>&1
for label in A B C; do
  SOURCE_DATE_EPOCH=1621202600 uv build --wheel --offline --no-build-isolation --python "$out/.build-venv/bin/python" --out-dir "$out/build-$label" "$out/source-$label/sortedcontainers-2.4.0" > "$out/build-$label.log" 2>&1
done

python3.12 - "$inputs" "$out" <<'PY'
import base64
import csv
import hashlib
import pathlib
import sys
import zipfile

inputs, out = map(pathlib.Path, sys.argv[1:])
wheels = [next((out / f'build-{label}').glob('*.whl')) for label in 'ABC']
assert wheels[1].read_bytes() == wheels[2].read_bytes()
with zipfile.ZipFile(inputs / 'sortedcontainers-2.4.0-py2.py3-none-any.whl') as official:
    with zipfile.ZipFile(wheels[0]) as baseline, zipfile.ZipFile(wheels[1]) as changed:
        for name in official.namelist():
            if name.startswith('sortedcontainers/') and name.endswith('.py'):
                assert baseline.read(name) == changed.read(name) == official.read(name), name
        original_license = 'sortedcontainers-2.4.0.dist-info/LICENSE'
        rebuilt_license = 'sortedcontainers-2.4.0.dist-info/licenses/LICENSE'
        assert baseline.read(rebuilt_license) == changed.read(rebuilt_license) == official.read(original_license)
        notice = 'sortedcontainers-2.4.0.dist-info/licenses/LICENSE-APACHE-2.0.txt'
        assert changed.read(notice) == (inputs / 'LICENSE-2.0.txt').read_bytes()
for wheel in wheels:
    with zipfile.ZipFile(wheel) as archive:
        record = next(name for name in archive.namelist() if name.endswith('.dist-info/RECORD'))
        for name, digest, size in csv.reader(archive.read(record).decode().splitlines()):
            if name == record:
                assert digest == size == ''
                continue
            data = archive.read(name)
            assert len(data) == int(size), name
            assert digest == 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), name
    print(wheel.relative_to(out), hashlib.sha256(wheel.read_bytes()).hexdigest())
PY
