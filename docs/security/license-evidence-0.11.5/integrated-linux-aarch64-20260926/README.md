# Integrated 0.11.5 Linux aarch64 wheel candidates

These are local candidates, not published PyPI artifacts or a release-wide verdict.

Source: integrated commit `68cb714303c79406c2b2fd6571a1308c9329aace`, Git archive SHA256
`0be425c5b82b7e0e9c3c30635273e2c78d5c87bb42c132c5919ee1c4083d74f8`.
The binding `Cargo.lock` SHA256 is
`528414b582b256a6ab8ac68ceceb134e8ddb719774513ab8cc3a0e4eb22a177d`.
After running `tools/select_third_party_license.sh aarch64-unknown-linux-gnu`,
the source differs from the archive only in `LICENSE-THIRD-PARTY` and
`tools/third_party_licenses.snapshot.json`. Their SHA256 values are respectively
`afa61d22b98ec3b8d4797c4af1dc9a6d159b2051a543e658c67f4d2404d2d87c`
and `23452c4e7daaf7b36cbf2036ab4d1395d106b7bc5cfcef2bb4572d24ac4b476b`.

The checked-in build scripts record the exact local commands. They used an ARM64
Podman VM, pinned `manylinux2014_aarch64` image digest
`d36e257b4f7b1130a1442a5cd28022307645a92b8093ab65543205426354cfd2`,
Rust 1.97.1, maturin 1.14.1, Cargo offline/locked, and
`SOURCE_DATE_EPOCH=1790068752`. The local build directory was
`/Users/seonghobae/orca/workspaces/fast-mlsirm/fmls-license-arm-native-20260926`.
The cached crates were checked against `Cargo.lock` by Cargo. The maturin ARM64
wheel used for the offline environment had SHA256
`a131d912b5267e640bc96d70f4914e10590aed64082ec9abacba7cea52004224`.

| CPython | Wheel SHA256 |
| --- | --- |
| 3.12 | `a663a645c241a3f13570ca017263ef857a11e856a1b240588fe93d74a41228bb` |
| 3.13 | `13f8a6b637cc96d51b45a32513a5857fdabbae02c5293f8a66c45a4a57df39a7` |
| 3.14 | `9bc29e9e07978b6fa0c1cc7a7ca220cb674f71a23ca39affb24ddb9e5c1e13a9` |

For each exact wheel, `own_crate_wheel_license_files` returned version 0.11.5,
six source-matching `METADATA License-File` members, and no errors. The native
extension is ELF AArch64 with the matching CPython tag. `auditwheel` reported
`manylinux_2_17_aarch64` and no external shared-library requirements. The
individual output is in `license-verify.log`, the three `auditwheel.log` files,
and `SHA256SUMS`. No installed-wheel regression suite was run on ARM64.

The earlier local cp312 build from integrated commit `1c08f1bc` had the same
wheel SHA256 as the cp312 rebuild above, but its selected source still had the
Linux x86_64 snapshot. The selector fix in [draft PR #2174](https://github.com/ContextualWisdomLab/fast-mlsirm/pull/2174)
now selects and checks the target snapshot with the notice; the rebuilt wheel
and its source pass the complete wheel-license check. The ten macOS `objc2`
HOLD rows and three Python HOLD rows remain outside this ARM64 result.
