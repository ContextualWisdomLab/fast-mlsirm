#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-license-integrated-41da-20260927
IMG=quay.io/pypa/manylinux2014_x86_64@sha256:21c37461985655aaa25ed3a923b28e6c9d4dd9e10edc0281eb50385184bddd31
export SOURCE_DATE_EPOCH=1790068752
test "$(sha256sum "$D/source.tar.gz" | cut -d' ' -f1)" = 15463221055c8c3866e51ed7c32ad8f7aa9b54a99cf2a4c9b1bdd6d10ba48c50
for ABI in cp313-cp313 cp314-cp314; do
  test ! -e "$D/core-dist-$ABI"
  mkdir -p "$D/core-dist-$ABI" "$D/core-logs-$ABI"
done
docker run --pull=never --network=none --rm --cpuset-cpus=0-7 --memory=24g -u "$(id -u):$(id -g)" \
  -e SOURCE_DATE_EPOCH -e CARGO_HOME=/cargo -e RUSTUP_HOME=/rustup -e CARGO_NET_OFFLINE=true \
  -e CARGO_BUILD_JOBS=8 -e MATURIN_NO_INSTALL_RUST=1 -e OPENBLAS_NUM_THREADS=1 \
  -e CARGO_TARGET_DIR=/w/core-target-manylinux2014 \
  -v "$D:/w" -v /home/seongho/.cargo:/cargo -v /home/seongho/.rustup:/rustup:ro \
  "$IMG" bash -euo pipefail -c '
    export HOME=/w/home
    export PATH=/rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin:/w/container-venv-2014build/bin:$PATH
    cd /w/core-src
    for ABI in cp313-cp313 cp314-cp314; do
      maturin build --release --offline --locked --compatibility manylinux2014 \
        -i /opt/python/$ABI/bin/python -o /w/core-dist-$ABI \
        > /w/core-logs-$ABI/core-build.log 2>&1
      auditwheel show /w/core-dist-$ABI/fast_mlsirm-*.whl \
        > /w/core-logs-$ABI/auditwheel.log 2>&1
    done
  '
for ABI in cp313-cp313 cp314-cp314; do
  sha256sum "$D"/core-dist-$ABI/*.whl "$D"/core-logs-$ABI/*.log
done
