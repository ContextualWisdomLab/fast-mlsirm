# Integrated 0.11.5 Linux wheel candidates, 2026-09-27

These six local candidates are not published PyPI files or a release-wide
license verdict. They were built from integrated commit
`41da2684315e6be1ce2fbb7e8c861c8b27313c65` (`git archive --format=tar.gz
HEAD` SHA256 `15463221055c8c3866e51ed7c32ad8f7aa9b54a99cf2a4c9b1bdd6d10ba48c50`).
The binding `crates/fast-mlsirm-py/Cargo.lock` SHA256 is
`528414b582b256a6ab8ac68ceceb134e8ddb719774513ab8cc3a0e4eb22a177d`.
`SOURCE_DATE_EPOCH=1790068752`, Rust 1.97.1, maturin 1.14.1, offline Cargo,
and pinned manylinux2014 images were used. The exact commands and logs are
beside this file. A few copied script comments retain older candidate labels;
the executed `D` paths and source SHA256 assertions bind this 41da source.
The source and wheels remain in these local directories:

- x86_64: `/data/orca/workspaces/fmls-license-integrated-41da-20260927/` on `s1.cluster.seonghobae.me`;
- aarch64: `/Users/seonghobae/orca/workspaces/fast-mlsirm/fmls-license-integrated-41da-20260927/`.

| Target | CPython | Wheel SHA256 |
| --- | --- | --- |
| Linux x86_64 | 3.12 | `5857e018cf4040e583f822786d520b6ed87322f23d3df37f16c8ed994627f0d6` |
| Linux x86_64 | 3.13 | `0e31da23ec42da997dcaace8d58607a539d84a2b639e59d4896cb3f6ebb05300` |
| Linux x86_64 | 3.14 | `542a4ee486557b0741646dcbb8bb7eb925267f6c3e13c65f2ace447b336a8a93` |
| Linux aarch64 | 3.12 | `a663a645c241a3f13570ca017263ef857a11e856a1b240588fe93d74a41228bb` |
| Linux aarch64 | 3.13 | `cde352d9c9df6596bdb7ec17e13f0aa8195c3206c16ef1e282b2ced0946915c0` |
| Linux aarch64 | 3.14 | `9bc29e9e07978b6fa0c1cc7a7ca220cb674f71a23ca39affb24ddb9e5c1e13a9` |

The selected x86_64 third-party notice and snapshot SHA256 values are
`d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0`
and `e6d920bd6bdd392cc902a5a54fe365f48fdadb1b3762e857fddee1e4dea6ca26`.
For aarch64 they are
`afa61d22b98ec3b8d4797c4af1dc9a6d159b2051a543e658c67f4d2404d2d87c`
and `23452c4e7daaf7b36cbf2036ab4d1395d106b7bc5cfcef2bb4572d24ac4b476b`.
Each exact wheel passed the release workflow's `verify_wheel_license.py`, ZIP
integrity, matching CPython tag and ELF architecture, and
`own_crate_wheel_license_files`: version 0.11.5, six source-identical
`METADATA License-File` members, no errors. `auditwheel` reported
`manylinux_2_17` for both architectures. The x86_64 CPython 3.12 installed
wheel passed ten focused regression tests; no installed-wheel regression suite
was run for the other five candidates.

The same integrated source regenerated unfiltered Cargo metadata and filtered
target graphs. The target metadata SHA256 values are
`e169bf0fd7e79dc74924a95c00e2783643b3e02aa1d7762ad4e5921b5ac4c9ef`
(x86_64) and `3f12aed3146f4dee3004f3c9b09e26e59bc33fa2afe0f5307347cf17aebbf76c`
(aarch64). Target-specific verifier reruns exited zero with no completeness
gaps. Their inventory SHA256 values are
`048bf9f19d310ecd0981b4ccce1c04029efd8fee53e428e4ea743630d6690358`
and `8d08bdef4cfde6553d26944d91d84b4c1537e3e9d76e4a321744d1f618d8e221`.
Each binding target graph has 103 PERMISSIVE rows and zero HOLD. Fresh
snapshots contain the same 101 third-party rows as JSON data as the
snapshots used by the wheels; only the recorded source-inventory digest changes.
The reruns reused the earlier CycloneDX wheel SBOM input (SHA256
`00f486568b3b9ef17f81e9990ddf445de39ae3f11bae75fafdb45bddba53dcce`)
for the unchanged binding crate set; it is not a newly generated SBOM for
these candidates.

Python HOLD remains three (`atheris`, `numpy`, `sortedcontainers`). These
target-specific reruns have 158 rows in the binding-lock union, of which eleven
remain HOLD: ten `objc2` family rows and `android_system_properties`, as in the
[union review](../nonlinux-upstream-20260926/REVIEW.md). The broader
`cargo_by_class` summary counts workspace and SBOM rows too (229 rows, 32
HOLD); it is not the 158-row binding union. Windows and macOS candidate wheels
and the published 12-wheel matrix have not been verified here. No final gate verdict follows
from this Linux-only evidence.

## Non-Linux source graph follow-up

The same extracted source was rerun with Cargo 1.97.1, `--locked --offline`,
and `--filter-platform x86_64-pc-windows-msvc`. The target metadata SHA256 is
`01360550ff35229a0452e49944e55d83f7cf39fa868b7c789e780f0dbf609690`.
The verifier inventory SHA256 is
`3495f86033b068e0fb4155cc38115caf2327b2d40804c859a86b3e66992c933b`
with exit code zero and no completeness gaps. Its 119 binding target rows are
117 PERMISSIVE third-party crates and two HOLD own crates. The own-crate HOLD
is expected: this graph rerun used the hash-bound Linux CPython 3.12 wheel,
whose third-party notice does not match the Windows graph. The newly generated
Windows snapshot SHA256 is
`834bbed269375c36f4b8ca867697c895e33b3f2a5b181accf728fb41593d22cd`;
its 117 row objects exactly match the committed Windows snapshot. These files
are under `inventory-windows-x86_64/` in the x86_64 build directory above.
This establishes the third-party source graph only. A Windows wheel built from
this source must still be checked against the Windows notice and source bytes.

The same `--locked --offline` rerun for `x86_64-apple-darwin` produced target
metadata SHA256 `98a6eb1bccfebf3c32d35c63df7041056a3c3c75efbe5ff2c0baeb8e3bbb502c`
and inventory SHA256 `918de6e15bd8a6e6697d11aee886c819346645cdba22ab7fb21cae160c4f40af`.
For `aarch64-apple-darwin` they are
`55f571392ae406d3740368d5c6c8d087bd178ffd9ac13367186dddcf6e5bbc1d`
and `393e48a3641bc0c57346cfe24bc78c18fdd7461ea9e5c897966f563a81fa65a7`.
Both verifier runs exited zero with no completeness gaps. Each target graph
has 114 rows: 102 PERMISSIVE, the same ten `objc2` family HOLD rows, and two
own-crate HOLD rows because the input is a Linux wheel with a Linux notice.
These inventories are under `inventory-<target>/` in the x86_64 build directory.
They do not verify a macOS wheel or clear its ten third-party HOLD rows.
