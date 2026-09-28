# Integrated 0.11.5 Linux cp312 candidate

Source: merge commit `1c08f1bcbd8f6d8be2b1b4f1be79640282dad787`
(parents `56d9b6b1` and `cf710bc1`), exported with `git archive`.
Source archive SHA256:
`7d8b1e1c917d6be5eeeed822742b8ec31add587b96ad62bfb68c49ceaaca1b01`.
Build host path:
`/data/orca/workspaces/fmls-a3-0115-integrated-1c08-20260926/` on
`s1.cluster.seonghobae.me`. The checked-in `build-core.sh` records the pinned
manylinux image digests, offline build inputs, and test command.

The fresh wheel
`fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
has SHA256
`d8ec1d497763abd943dfc5ba0defa93a67f141b8bab9adf04a02c8d09a9d43bb`.
It is byte-identical to the earlier A3 wheel from source `58b7b23f`.
`auditwheel` reports `manylinux_2_17_x86_64`; the installed wheel passed
10 regression tests with NumPy 2.5.2. Against the extracted integrated source,
the explicit actual-wheel license check exited 0 and matched all six source
license files. `tools/verify_wheel_license.py` also found the wheel's
`LICENSE-THIRD-PARTY` SHA256
`d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0`.

This proves the locally built integrated Linux cp312 candidate only. It does
not establish the other eleven wheel artifacts or a published release gate.
