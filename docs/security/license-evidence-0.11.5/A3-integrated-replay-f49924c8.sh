#!/usr/bin/env bash
set -u
e=/data/orca/workspaces/fmls-license-evidence
r=$e/runs/fmls-a3-integrated-f49924c8-20260926
c=/data/orca/workspaces/fmls-a3-0115-reviewfix-58b7b23f/core-src
i=$e/py-exact-0115-20260926/inputs
m=$e/runs/fmls-own-wheel-0115-20260926/inputs
w=/data/orca/workspaces/fmls-a3-0115-reviewfix-58b7b23f/core-dist-manylinux2014/fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
cd "$e" || exit 2
"$e/venv312/bin/python" "$r/license_inventory.py" \
  --cargo-metadata-workspace "$m/ws.json" \
  --cargo-metadata-binding "$m/py.json" \
  --cargo-metadata-binding-target "$m/binding-x86_64-unknown-linux-gnu.json" \
  --binding-target x86_64-unknown-linux-gnu \
  --cargo-lock-workspace "$c/Cargo.lock" \
  --cargo-lock-binding "$c/crates/fast-mlsirm-py/Cargo.lock" \
  --cargo-registry-cache "$e/cargo-home/registry/cache" \
  --wheel-sbom "$m/wheel-sbom.json" --tree-dir "$m/tree" \
  --uv-lock "$c/uv.lock" \
  --requirements "$c/requirements/ci.txt" "$c/requirements/package.txt" \
  --extra-python 'maturin==1.14.1:publish-pypi.yml maturin-action binary' \
  --python-scope "$i/python-scope.json" \
  --pypi-meta-dir "$i/pypi-meta" --pypi-artifact-dir "$i/pypi-art" \
  --cargo-upstream-license-evidence "$e/runs/fmls-cfg-aliases-0115-20260926/upstream-evidence.json" \
  --own-crate-wheel "$w" \
  --own-crate-wheel-sha256 d8ec1d497763abd943dfc5ba0defa93a67f141b8bab9adf04a02c8d09a9d43bb \
  --own-crate-wheel-source-root "$c" \
  --reviewed-pointer-notices \
  --out "$r/inventory.json" >"$r/stdout.log" 2>"$r/stderr.log"
rc=$?
printf '%s\n' "$rc" >"$r/exit_code"
exit "$rc"
