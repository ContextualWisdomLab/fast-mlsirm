#!/bin/bash
# usage: rerun-py-exact.sh <outdir> [extra args]: 0.11.5 Cargo inputs + 26cd4c83 Python inputs (inputs/), verifier 8f8916e5
set -u
D=/data/orca/workspaces/fmls-license-integrated-41da-20260927; E=/data/orca/workspaces/fmls-license-evidence; P=$E/py-exact-0115-20260926; I=$P/inputs; CI=$E/runs/fmls-own-wheel-0115-20260926/inputs
W=$D/core-dist-manylinux2014/fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
mkdir -p "$1"; cd "$E" || exit 2
"$E/venv312/bin/python" $D/core-src/tools/license_inventory.py --cargo-metadata-workspace $D/workspace.json --cargo-metadata-binding $D/binding.json \
  --cargo-lock-workspace $D/core-src/Cargo.lock --cargo-lock-binding $D/core-src/crates/fast-mlsirm-py/Cargo.lock --cargo-registry-cache cargo-home/registry/cache \
  --wheel-sbom $CI/wheel-sbom.json --tree-dir $CI/tree --uv-lock $I/uv.lock --requirements $I/requirements/ci.txt $I/requirements/package.txt \
  --extra-python 'maturin==1.14.1:publish-pypi.yml maturin-action binary' --python-scope $I/python-scope.json \
  --pypi-meta-dir $I/pypi-meta --pypi-artifact-dir $I/pypi-art \
  --cargo-upstream-license-evidence $E/nonlinux-upstream-20260926/upstream-evidence.json --own-crate-wheel $W --own-crate-wheel-sha256 5857e018cf4040e583f822786d520b6ed87322f23d3df37f16c8ed994627f0d6 --own-crate-wheel-source-root $D/core-src --reviewed-pointer-notices \
  "${@:2}" --out "$1/inventory.json" > "$1/stdout.log" 2> "$1/stderr.log"
echo $? > "$1/exit_code"
