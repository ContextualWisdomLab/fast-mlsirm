# Integrated 0.11.5 Linux cp313 and cp314 candidates

Source: merge commit `1c08f1bcbd8f6d8be2b1b4f1be79640282dad787`,
Git archive SHA256
`7d8b1e1c917d6be5eeeed822742b8ec31add587b96ad62bfb68c49ceaaca1b01`.
The checked-in `build-py313-314.sh` uses the same pinned, offline
manylinux2014 image and Rust toolchain as the [cp312 build](../integrated-linux-cp312-20260926/README.md).
The build host directory is
`/data/orca/workspaces/fmls-a3-0115-integrated-1c08-20260926/` on
`s1.cluster.seonghobae.me`.

| Wheel | SHA256 | `auditwheel` platform |
| --- | --- | --- |
| `fast_mlsirm-0.11.5-cp313-cp313-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `0e31da23ec42da997dcaace8d58607a539d84a2b639e59d4896cb3f6ebb05300` | `manylinux_2_17_x86_64` |
| `fast_mlsirm-0.11.5-cp314-cp314-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `542a4ee486557b0741646dcbb8bb7eb925267f6c3e13c65f2ace447b336a8a93` | `manylinux_2_17_x86_64` |

For each exact wheel, all six `METADATA License-File` members were present,
their bytes matched the extracted integrated source, and the matching
CPython extension was present. `own_crate_wheel_license_files` returned
version 0.11.5, six bound files, and no errors. The bundled
`LICENSE-THIRD-PARTY` SHA256 in both is
`d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0`.
The six individual file hashes are listed in the [A3 evidence](../A3-license-gate-verdict-20260926.md#actual-wheel-correction-and-binding).

These are locally built candidates. No installed-wheel regression test was
run for cp313 or cp314, and neither is a published artifact verdict.
