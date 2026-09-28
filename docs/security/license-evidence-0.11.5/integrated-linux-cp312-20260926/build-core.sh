#!/usr/bin/env bash
# Integrated candidate: isolated core-only cp312 build of exact merge source 1c08f1bc, retaining complete libm notices
# in pinned manylinux2014 (glibc 2.17). Build+auditwheel in 2014 image; installed-wheel regression
# in the same pinned 2_28 image as the validated pair (NumPy candidate needs glibc >= 2.27).
set -euo pipefail
D=/data/orca/workspaces/fmls-a3-0115-integrated-1c08-20260926
IMG14=quay.io/pypa/manylinux2014_x86_64@sha256:21c37461985655aaa25ed3a923b28e6c9d4dd9e10edc0281eb50385184bddd31
IMG28=quay.io/pypa/manylinux_2_28_x86_64@sha256:ae21cd1c8220f773f9b5934f3b845d677494d4cd4b8981c1e416f654e8afa74e
export SOURCE_DATE_EPOCH=1790068752
test ! -e "$D/core-tpl.exit"   # one attempt only
test "$(sha256sum "$D/source.tar.gz" | cut -d' ' -f1)" = 7d8b1e1c917d6be5eeeed822742b8ec31add587b96ad62bfb68c49ceaaca1b01
mkdir -p "$D/core-dist-manylinux2014" "$D/core-logs-2014"
L=/w/core-logs-2014
COMMON=(--pull=never --network=none --rm --cpuset-cpus=0-7 --memory=24g -u "$(id -u):$(id -g)"
  -e SOURCE_DATE_EPOCH -e CARGO_HOME=/cargo -e RUSTUP_HOME=/rustup -e CARGO_NET_OFFLINE=true
  -e CARGO_BUILD_JOBS=8 -e MATURIN_NO_INSTALL_RUST=1 -e OPENBLAS_NUM_THREADS=1 -e RAYON_NUM_THREADS=2
  -e PYTHONHASHSEED=0 -v "$D:/w" -v /home/seongho/.cargo:/cargo -v /home/seongho/.rustup:/rustup:ro)

echo "stage=build $(date -u +%FT%TZ)" > "$D/core-2014.stage"
docker run "${COMMON[@]}" -e CARGO_TARGET_DIR=/w/core-target-manylinux2014 "$IMG14" bash -euo pipefail -c '
  export HOME=/w/home L='"$L"'
  export PATH=/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:/w/container-venv-2014build/bin:$PATH
  ldd --version | sed -n 1p > $L/glibc.txt; gcc --version | sed -n 1p >> $L/glibc.txt; rustc --version >> $L/glibc.txt
  /opt/python/cp312-cp312/bin/python -m venv /w/container-venv-2014build
  /w/container-venv-2014build/bin/python -m pip install --no-index --find-links=/w/core-build-deps maturin==1.14.1 > $L/tool-install.log 2>&1
  cd /w/core-src
  maturin build --release --offline --locked --compatibility manylinux2014 -i /w/container-venv-2014build/bin/python -o /w/core-dist-manylinux2014 > $L/core-build.log 2>&1
  auditwheel show /w/core-dist-manylinux2014/fast_mlsirm-*.whl > $L/auditwheel.log 2>&1
'
echo "stage=regression $(date -u +%FT%TZ)" > "$D/core-2014.stage"
docker run "${COMMON[@]}" "$IMG28" bash -euo pipefail -c '
  export HOME=/w/home L='"$L"'
  /opt/python/cp312-cp312/bin/python -m venv /w/container-venv-2014test
  /w/container-venv-2014test/bin/python -m pip install --no-index --find-links=/w/core-build-deps pytest==9.1.1 > $L/test-tool-install.log 2>&1
  /w/container-venv-2014test/bin/python -m pip install --no-index --no-deps /w/wheel-noticed/numpy-2.5.2-*.whl > $L/numpy-install.log 2>&1
  /w/container-venv-2014test/bin/python -m pip install --no-index --no-deps /w/core-dist-manylinux2014/fast_mlsirm-*.whl > $L/core-install.log 2>&1
  cd /w
  /w/container-venv-2014test/bin/python -m pytest -c /dev/null /w/core-src/tests/test_rust_regression_ols_hc_parity.py -q -vv -p no:cacheprovider > $L/regression.log 2>&1
'
echo "stage=done $(date -u +%FT%TZ)" > "$D/core-2014.stage"
sha256sum "$D"/core-dist-manylinux2014/*.whl "$D"/core-logs-2014/*
