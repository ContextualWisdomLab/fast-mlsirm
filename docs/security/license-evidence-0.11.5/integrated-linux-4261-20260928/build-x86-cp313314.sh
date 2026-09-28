#!/usr/bin/env bash
set -euo pipefail

D=/data/orca/workspaces/fmls-a3-4261-linux-x86-cp313314-20260928
P=/data/orca/workspaces/fmls-a3-integrated-4261e8e6-20260928
IMG=quay.io/pypa/manylinux2014_x86_64@sha256:21c37461985655aaa25ed3a923b28e6c9d4dd9e10edc0281eb50385184bddd31
SOURCE_SHA=eb98cb0272dedf2a758e3a172b48bd5b63cf98251be680ff387e79473f0c977a

test ! -e "$D"
mkdir -p "$D/src" "$D/deps" "$D/logs-cp313" "$D/logs-cp314" "$D/dist-cp313" "$D/dist-cp314" "$D/home"
cp "$P/source.tar.gz" "$D/source.tar.gz"
test "$(sha256sum "$D/source.tar.gz" | cut -d' ' -f1)" = "$SOURCE_SHA"
tar -xzf "$D/source.tar.gz" -C "$D/src"
cp "$P/core-build-deps/maturin-1.15.0-"*.whl "$D/deps/"
cp -a "$P/core-target-manylinux2014" "$D/target"

export SOURCE_DATE_EPOCH=1790553578
docker run --pull=never --network=none --rm --cpuset-cpus=8-9 --memory=6g \
  -u "$(id -u):$(id -g)" -e SOURCE_DATE_EPOCH -e CARGO_HOME=/cargo \
  -e RUSTUP_HOME=/rustup -e CARGO_NET_OFFLINE=true -e CARGO_BUILD_JOBS=2 \
  -e MATURIN_NO_INSTALL_RUST=1 -e OPENBLAS_NUM_THREADS=1 \
  -e CARGO_TARGET_DIR=/w/target \
  -v "$D:/w" -v /home/seongho/.cargo:/cargo -v /home/seongho/.rustup:/rustup:ro \
  "$IMG" bash -euo pipefail -c '
    export HOME=/w/home
    export PATH=/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:/w/venv/bin:$PATH
    /opt/python/cp312-cp312/bin/python -m venv /w/venv
    /w/venv/bin/python -m pip install --no-index --find-links=/w/deps maturin==1.15.0 > /w/tool-install.log 2>&1
    cd /w/src
    bash tools/select_third_party_license.sh x86_64-unknown-linux-gnu
    for PY in 313 314; do
      maturin build --release --offline --locked --compatibility manylinux2014 \
        -i /opt/python/cp${PY}-cp${PY}/bin/python -o /w/dist-cp${PY} \
        > /w/logs-cp${PY}/build.log 2>&1
      /opt/python/cp312-cp312/bin/python tools/verify_wheel_license.py /w/dist-cp${PY} \
        > /w/logs-cp${PY}/license.log 2>&1
      auditwheel show /w/dist-cp${PY}/fast_mlsirm-*.whl \
        > /w/logs-cp${PY}/auditwheel.log 2>&1
    done
  '
sha256sum "$D"/dist-cp313/*.whl "$D"/dist-cp314/*.whl
