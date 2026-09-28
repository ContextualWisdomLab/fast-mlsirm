#!/usr/bin/env bash
set -euo pipefail
D=/Users/seonghobae/orca/workspaces/fast-mlsirm/fmls-license-arm-native-20260926
IMG=quay.io/pypa/manylinux2014_aarch64@sha256:d36e257b4f7b1130a1442a5cd28022307645a92b8093ab65543205426354cfd2
export SOURCE_DATE_EPOCH=1790068752
test "$(shasum -a 256 "$D/source-reviewed.tar.gz" | cut -d' ' -f1)" = 0be425c5b82b7e0e9c3c30635273e2c78d5c87bb42c132c5919ee1c4083d74f8
(cd "$D/arm-src" && bash tools/select_third_party_license.sh aarch64-unknown-linux-gnu)
test "$(shasum -a 256 "$D/arm-src/tools/third_party_licenses.snapshot.json" | cut -d' ' -f1)" = 23452c4e7daaf7b36cbf2036ab4d1395d106b7bc5cfcef2bb4572d24ac4b476b
test "$(shasum -a 256 "$D/arm-src/LICENSE-THIRD-PARTY" | cut -d' ' -f1)" = afa61d22b98ec3b8d4797c4af1dc9a6d159b2051a543e658c67f4d2404d2d87c
test ! -e "$D/dist-reviewed-cp314"
mkdir -p "$D/dist-reviewed-cp314" "$D/logs-reviewed-cp314" "$D/home"
docker volume inspect fmls-arm-target-1c08-20260926 >/dev/null 2>&1 || docker volume create fmls-arm-target-1c08-20260926 >/dev/null
docker run --platform linux/arm64 --pull=never --network=none --rm --memory=5g \
  -e SOURCE_DATE_EPOCH -e CARGO_HOME=/w/cargo -e CARGO_NET_OFFLINE=true \
  -e CARGO_BUILD_JOBS=3 -e MATURIN_NO_INSTALL_RUST=1 -e CARGO_TARGET_DIR=/target \
  -v "$D:/w" -v fmls-arm-target-1c08-20260926:/target \
  "$IMG" bash -euo pipefail -c '
    export HOME=/w/home
    export PATH=/w/rustup-arm/toolchains/1.97.1-aarch64-unknown-linux-gnu/bin:/w/venv/bin:$PATH
    ldd --version | sed -n 1p > /w/logs-reviewed-cp314/toolchain.txt
    rustc --version >> /w/logs-reviewed-cp314/toolchain.txt
    maturin --version >> /w/logs-reviewed-cp314/toolchain.txt
    cd /w/arm-src
    maturin build --release --offline --locked --compatibility manylinux2014 \
      -i /opt/python/cp314-cp314/bin/python -o /w/dist-reviewed-cp314 \
      > /w/logs-reviewed-cp314/core-build.log 2>&1
    auditwheel show /w/dist-reviewed-cp314/fast_mlsirm-*.whl \
      > /w/logs-reviewed-cp314/auditwheel.log 2>&1
  '
shasum -a 256 "$D"/dist-reviewed-cp314/*.whl "$D"/logs-reviewed-cp314/*
