# FIPC GIL regression probe lifecycle

Implementation tested: `1812cdcb7c1557dd5065c523007498fcd733eb9b`,
a test-only follow-up to #2285 (`5d10ac1df5deaf39bb7f0497bc74d23c38fe3973`).
No estimator, numerical criterion or binding change in this follow-up.

## Root cause and correction

The original observer signaled completion only after a successful fit. A worker
exception left the main thread spinning indefinitely, accumulating timestamp
objects without a bound. An owned subprocess fault reproduction entered the
injected worker failure and exceeded its 3-second deadline.

The observer now signals completion in `finally`, propagates worker exceptions,
uses constant-size maximum-gap state and cooperative sampling, and bounds its
join and observation deadline. The actual numerical fixture, including warm-up,
runs in an owned subprocess with a separate 90-second timeout. This external
bound is necessary because Python cannot interrupt a native call that holds the
GIL forever. The existing GIL acceptance criterion is unchanged.

Five focused lifecycle/control tests passed before native rebuild. They cover
worker failure, deadline, constant-size timing, GIL-held Python control, and the
external subprocess timeout. They are not numerical performance evidence.

## Native validation and resource receipt

The existing extension failed the bounded actual probe: the main thread froze
for 0.148 seconds of a 0.148-second fit. The child reported its extension hash:
`5d1258be791f499370832348ec728ea61aed6d9eb11b44477237e03f6a4f5b20`.
That artifact was backed up before replacement; its compiled-source identity
was not established merely by observing its pathname or hash.

A single approved offline editable build of the implementation head above ran
on macOS arm64 / CPython 3.14 with the installed pinned Rust 1.97.1 toolchain,
uv 0.12.3 and `pyo3/extension-module`. Actual build interval:
**2026-09-30 18:19:19.790–18:21:32.786 KST**, exit 0, 133.01 seconds.

```sh
CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true MATURIN_NO_INSTALL_RUST=1 \
UV_OFFLINE=1 CARGO_TARGET_DIR="$PWD/.venv/gil-lifecycle/target-isolated" \
nice -n 10 uv pip install --offline --no-deps --python .venv/bin/python -e .
```

The build used an owner-private cold target, one compilation job, a 600-second
process-group deadline, bounded output and resource abort guards. No shared
cache cleanup, toolchain provisioning or network dependency fallback occurred.
Observed minima: swap free 583.12 MiB, memory free 28%; neither crossed the
512 MiB / 25% abort guards. No second build was attempted.

After successful build, the same supported checkout imported extension hash
`8c7930d93203abd01de55870e309f1a42a592405645e8c2ce8cde8d0d184d00e`.
The numerical GIL probe and lifecycle tests then passed:

```sh
.venv/bin/python -m pytest tests/test_gil_detach_fipc.py \
  tests/test_gil_detach_fipc_lifecycle.py -q -p no:cacheprovider
```

**6 passed, exit 0**, 0.89 seconds; actual test interval
**2026-09-30 18:22:37.296–18:22:38.449 KST**. No full suite or GPU run.

Local detailed receipts are retained under the supported checkout's ignored
`.venv/gil-lifecycle/`: `before-build.json`, `build.json`, `build.log`,
`native-before.txt`, `after-build.json` and `native-after.txt`.
This is local current-tree build/test evidence, not hosted exact-head approval,
installed-wheel release acceptance or independent worker attestation.
