#!/usr/bin/env bash
set -euo pipefail

D=/data/orca/workspaces/fmls-a3-4261-linux-x86-cp313314-20260928
P=/data/orca/workspaces/fmls-a3-integrated-7e1f6c14-20260926
IMG=quay.io/pypa/manylinux_2_28_x86_64@sha256:ae21cd1c8220f773f9b5934f3b845d677494d4cd4b8981c1e416f654e8afa74e
test -d "$D/dist-cp313" && test -d "$D/dist-cp314"
test ! -e "$D/installed-tests.done"
mkdir -p "$D/test-deps"

python3 - "$D/test-deps" <<'PY'
import hashlib
import pathlib
import sys
import urllib.request

directory = pathlib.Path(sys.argv[1])
wheels = (
    ("https://files.pythonhosted.org/packages/7b/44/59a1eb68e773c4098d107ef34a0dbdeca501d72ffcfbff9a7707343921ce/numpy-2.5.2-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "29b86ff8a6cc556b47ec6b64b194815cc80e6bf5eedcc6cddfd65318cb0b4eee"),
    ("https://files.pythonhosted.org/packages/c7/99/461bd36dbdfac6c1c53efa370bd55a83227542d0d118f1677dbf1a3dacd5/numpy-2.5.2-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl", "318b9a4c845dbea06708a29c84ee429cc3065048db34cdb799047643492050ee"),
)
for url, expected in wheels:
    path = directory / url.rsplit("/", 1)[1]
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            path.with_suffix(".part").write_bytes(response.read())
        path.with_suffix(".part").rename(path)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"hash mismatch for {path.name}: {actual}")
    print(path.name, actual)
PY

docker run --pull=never --network=none --rm --cpuset-cpus=8-9 --memory=6g \
  -u "$(id -u):$(id -g)" -v "$D:/w" -v "$P/core-build-deps:/pytest-deps:ro" \
  "$IMG" bash -euo pipefail -c '
    export HOME=/w/home
    for PY in 313 314; do
      V=/w/venv-test-cp${PY}
      /opt/python/cp${PY}-cp${PY}/bin/python -m venv "$V"
      "$V/bin/python" -m pip install --no-index --no-deps /w/test-deps/numpy-2.5.2-cp${PY}-*.whl \
        > /w/logs-cp${PY}/numpy-install.log 2>&1
      "$V/bin/python" -m pip install --no-index --no-deps /w/dist-cp${PY}/fast_mlsirm-*.whl \
        > /w/logs-cp${PY}/core-install.log 2>&1
      "$V/bin/python" -m pip install --no-index --find-links=/pytest-deps pytest==9.1.1 \
        > /w/logs-cp${PY}/pytest-install.log 2>&1
      cd /w
      "$V/bin/python" -m pytest -c /dev/null /w/src/tests/test_rust_regression_ols_hc_parity.py \
        -q -vv -p no:cacheprovider > /w/logs-cp${PY}/installed-regression.log 2>&1
    done
  '
printf 'passed\n' > "$D/installed-tests.done"
