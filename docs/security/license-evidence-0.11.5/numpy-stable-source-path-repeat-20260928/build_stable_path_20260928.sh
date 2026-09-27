#!/usr/bin/env bash
# Candidate: numpy 2.5.2 against Fortran-free (NOFORTRAN=1 -> C_LAPACK) OpenBLAS 0.3.34,
# mirroring MacPython/openblas-libs x86_64 flags. Run label $1 (E/F) for a determinism pair.
set -euo pipefail
RUN=${1:?E or F required}
case "$RUN" in E|F) ;; *) echo "only fresh E/F labels are allowed" >&2; exit 1 ;; esac
ROOT=/data/orca/workspaces/fmls-numpy-pinned-backend-build-20260928
D="$ROOT/repeat_$RUN"
L=stable
(cd "$ROOT" && sha256sum --check input-sha256.txt)
mkdir "$D"
for input_dir in input src sdist; do ln -s "$ROOT/$input_dir" "$D/$input_dir"; done
IMG=quay.io/pypa/manylinux_2_28_x86_64@sha256:ae21cd1c8220f773f9b5934f3b845d677494d4cd4b8981c1e416f654e8afa74e
export SOURCE_DATE_EPOCH=1790068752
for output in "$D/b_$L" "$D/obl_$L" "$D/wh_$L" "$D/rep_$L"; do
  test ! -e "$output" || { echo "refusing existing output: $output" >&2; exit 1; }
done
mkdir "$D/b_$L" "$D/wh_$L" "$D/rep_$L"
tar xzf $D/src/OpenBLAS-0.3.34.tar.gz -C $D/b_$L
tar xzf $D/sdist/numpy-2.5.2.tar.gz -C $D/b_$L
docker run --rm --pull=never --network=none --cpus=2 --memory=6g -u "$(id -u):$(id -g)" -e XDG_CACHE_HOME=/w/b_$L/cache -e SOURCE_DATE_EPOCH -v $D:/w -v $ROOT/input:/w/input:ro -v $ROOT/src:/w/src:ro -v $ROOT/sdist:/w/sdist:ro $IMG nice -n 19 bash -euo pipefail -c "
  FLAGS=\"BUFFERSIZE=20 DYNAMIC_ARCH=1 USE_OPENMP=0 NUM_THREADS=64 BINARY=64 TARGET=PRESCOTT INTERFACE64=1 SYMBOLSUFFIX=64_ LIBNAMESUFFIX=64_ SYMBOLPREFIX=scipy_ LIBNAMEPREFIX=scipy_ FIXED_LIBNAME=1 NOFORTRAN=1 C_LAPACK=1 FC=/nonexistent-fortran-compiler\"
  cd /w/b_$L/OpenBLAS-0.3.34
  echo \"== openblas start \$(date +%T)\"; gcc --version | head -1
  CFLAGS=\"-fvisibility=protected -Wno-uninitialized\" make -j2 \$FLAGS DYNAMIC_LIST=\"PRESCOTT NEHALEM SANDYBRIDGE HASWELL SKYLAKEX\" > /w/b_$L/openblas-make.log 2>&1
  make \$FLAGS DYNAMIC_LIST=\"PRESCOTT NEHALEM SANDYBRIDGE HASWELL SKYLAKEX\" PREFIX=/w/obl_$L install > /w/b_$L/openblas-install.log 2>&1
  mv /w/obl_$L/lib/pkgconfig/openblas*.pc /w/obl_$L/lib/pkgconfig/scipy-openblas.pc
  sed -i -e \"s/\\(^Cflags.*\\)/\\1 -DBLAS_SYMBOL_PREFIX=scipy_ -DBLAS_SYMBOL_SUFFIX=64_/\" /w/obl_$L/lib/pkgconfig/scipy-openblas.pc
  # Match the official scipy-openblas64 pkg-config: the generated Libs line doubles the 64_ suffix.
  sed -i -e \"s|^Libs: .*|Libs: -L\\\${libdir} -lscipy_openblas64_|\" -e \"s/^\\(Cflags.*\\)/\\1 -DHAVE_BLAS_ILP64 -DOPENBLAS_ILP64_NAMING_SCHEME/\" /w/obl_$L/lib/pkgconfig/scipy-openblas.pc
  cat /w/obl_$L/lib/pkgconfig/scipy-openblas.pc
  PKG_CONFIG_PATH=/w/obl_$L/lib/pkgconfig pkg-config --libs --cflags scipy-openblas
  echo \"== numpy start \$(date +%T)\"
  export PKG_CONFIG_PATH=/w/obl_$L/lib/pkgconfig
  /opt/python/cp312-cp312/bin/python -m venv /w/b_$L/backend
  /w/b_$L/backend/bin/python -m pip install --no-cache-dir --no-index --find-links=/w/input/build-deps --require-hashes -r /w/input/build-requirements.txt > /w/b_$L/backend-install.log 2>&1
  export PATH=/w/b_$L/backend/bin:\$PATH
  /w/b_$L/backend/bin/python -m pip freeze --all > /w/b_$L/backend-freeze.txt
  /w/b_$L/backend/bin/python -m pip wheel -v /w/b_$L/numpy-2.5.2 --no-build-isolation --no-deps --no-index --no-cache-dir -w /w/wh_$L \
     -Csetup-args=-Duse-ilp64=true -Csetup-args=-Dallow-noblas=false -Cbuild-dir=/w/b_$L/npbuild -Ccompile-args=-j2 > /w/b_$L/numpy-build.log 2>&1
  grep -E \"Successfully installed|BLAS symbol suffix|Run-time dependency scipy-openblas|Library scipy-openblas\" /w/b_$L/numpy-build.log | head
  echo \"== repair start \$(date +%T)\"; auditwheel --version
  LD_LIBRARY_PATH=/w/obl_$L/lib auditwheel repair --plat manylinux_2_28_x86_64 -w /w/rep_$L /w/wh_$L/*.whl 2>&1 | tail -3
  echo \"== done \$(date +%T)\"
"
sha256sum $D/obl_$L/lib/*.so* $D/wh_$L/*.whl $D/rep_$L/*.whl
