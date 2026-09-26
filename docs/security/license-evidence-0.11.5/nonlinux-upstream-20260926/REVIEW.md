# Non-Linux Cargo license review — 2026-09-26

Status: **partial**. Coordinator-style technical review delegated by the maintainer; not legal review or a release gate verdict.

The input is the 0.11.5 union inventory from `rerun-785e675c` (SHA256 `f079cf2bf0d341a31a575a3672923965c5a3f7a795a32be8db1eb37b25ac5186`). Every registry archive was bound to its `Cargo.lock` SHA256. For crates without an archive license file, [`upstream-evidence.json`](upstream-evidence.json) binds the local text to the archive's `.cargo_vcs_info.json` commit and path; [`SOURCES.md`](SOURCES.md) records the raw URL and SHA256 of each file.

## Six distinct unverified archive texts

| Text (raw SHA256 prefix) | Rows | Finding | Decision |
| --- | ---: | --- | --- |
| `android_system_properties` `LICENSE-APACHE` (`216486f29671`) | 1 | Apache header only, without the complete license text. Its separate MIT file is verified. | HOLD: the all-candidate rule still applies. |
| `glutin_wgl_sys` `LICENSE` (`a44454274400`) | 1 | Complete Apache-2.0 text. Against the canonical Apache text, only the appendix's bracket notation (`[]` to `{}`) and its filled copyright placeholder differ. | Accepted by exact normalized hash, not by an SPDX-template claim. |
| `jni-sys` `LICENSE-APACHE` (`c6596eb7be85`) | 2 | Complete Apache-2.0 text. Only the appendix's bracket notation and placeholder differ; the same bytes are in both locked versions. | Accepted by exact normalized hash. This also resolves `jni-sys-macros` with the exact upstream file. |
| `objc2-foundation` `src/copying.rs` (`259b22f571b5`) | 1 | Rust source code, not a license document. | Ignore source files ending in `.rs` unless Cargo explicitly declares one as its license file. The crate still stays HOLD on its upstream notice. |
| `objc2-foundation` `src/tests/copying.rs` (`23d608079a47`) | 1 | Rust test code, not a license document. | Same as above. |
| Windows family `license-mit` (`c2cfccb812fe`) | 12 | After the title and copyright line, the full MIT body differs only by its missing final period. SPDX normally requires punctuation to match. | Accepted by exact normalized hash as a recorded individual judgment, not SPDX equivalence. |

The three accepted archive files are preserved in `files/artifact__*`; the fixture stores their complete bytes, archive hashes, file hashes, and normalized hashes. The Apache comparison used the canonical text in `/usr/share/common-licenses/Apache-2.0` and the [SPDX v3.29.0 Apache template](https://github.com/spdx/license-list-data/blob/31ba1a50e5397e00a304dbadc76531740e89ee48/template/Apache-2.0.template.txt) (SHA256 `5567209d3a634ebd6ed122a7302563b9668ab141941149a4e30d9c7b887f41b9`). [SPDX's matching guidelines](https://spdx.github.io/spdx-spec/v2.3.1-dev/license-matching-guidelines-and-templates/) require punctuation unless a stated template rule permits a variation.

## Thirteen archives without license files

- `gl_generator` and `khronos_api`: the full Apache text from their exact `gl-rs` publication commits matches a previously reviewed canonical hash. Both now pass.
- `ndk-sys`: the exact upstream MIT and Apache files both match previously reviewed hashes. It now passes.
- `jni-sys-macros`: the upstream MIT file matches a reviewed hash; its Apache file is the separately reviewed appendix variant above. It now passes.
- `block2`, `dispatch2`, `objc2`, `objc2-core-foundation`, `objc2-core-graphics`, `objc2-encode`, `objc2-io-surface`, `objc2-metal`, and `objc2-quartz-core`: exact publication commits all contain the same `LICENSE.md`. It states which crates have MIT or a Zlib/Apache/MIT choice, but does not contain the full grants and flags an [Apple SDK licensing caveat](https://github.com/madsmtm/objc2/blob/8852b424193ca41602281b3d7540d7c8ed51e49a/LICENSE.md). All nine stay HOLD. After removing the false `copying.rs` candidates, `objc2-foundation` binds to the same upstream notice and also stays HOLD.

## Re-run

The verifier at this branch head was run against the same 0.11.5 inputs with the new upstream manifest. Exit codes were zero, completeness gaps were empty, and Python HOLD stayed at 8.

| Graph | Before | After | Inventory SHA256 after |
| --- | ---: | ---: | --- |
| Cargo union, 158 rows | HOLD 30 | **HOLD 11** | `b7346b04777f5ed7fc3ee245f848b405739af871270eb86cd0e629373cdcf63e` |
| Linux x86_64, 103 rows | HOLD 0 | **HOLD 0** | `f73f958c89df8b3550a20a62967e93015c99dbafa29abd2e4052671915cb499f` |

The 19 changed verdicts are the 12 Windows rows, `glutin_wgl_sys`, `jni-sys` ×2, `jni-sys-macros`, `gl_generator`, `khronos_api`, and `ndk-sys`. The remaining 11 are the ten `objc2` family rows above plus `android_system_properties`. Their separate text or publication evidence needs a decision before a non-Linux gate can pass; this review does not turn a pointer or Apache header into a full license text.

Reproduction inputs and outputs on s1: `/data/orca/workspaces/fmls-license-evidence/nonlinux-upstream-20260926/` (`rerun-current.sh`, `current-union/`, `current-linux/`).

## Target-filtered binding graph (2026-09-26 follow-up)

The locked 0.11.5 binding manifest was resolved offline with Cargo 1.97.1 using
`cargo metadata --offline --locked --format-version 1 --filter-platform <triple>`.
Each JSON input and verifier result is retained beside the rerun script on s1.
The counts below include only rows in that target's binding resolve graph;
the inventory's unfiltered `cargo_by_class` summary also includes workspace rows.

| Target triple | Binding rows | Permissive | HOLD | Metadata SHA256 | Inventory SHA256 |
| --- | ---: | ---: | ---: | --- | --- |
| `x86_64-apple-darwin` | 114 | 104 | 10 | `bf0bff04e9c9a9324db2d05dd3bdcd6ed7c81e93755e5161e09fb008b1a5931d` | `6d964bdcb7fe95ab30df9b4af580a39d05438425e38cb0ac27da4a2fb888bc92` |
| `aarch64-apple-darwin` | 114 | 104 | 10 | `1181921e0de383aa2a1db14d9823e40f790f88f2518e28c7f1ffa8d0fe147398` | `7b8a8ffee1ce7d544ad84dd14de1773a3d2660171805a6babc8ab000de24d757` |
| `x86_64-pc-windows-msvc` | 119 | 119 | 0 | `87d3843c75d371a28d6bc5ad0370de76b2fa0b6a5a2e5f1600ff67025dd7734e` | `55dc00adb5c28086d726e1d1d74749da8021cee7bf89b224e2da2660d78e108d` |
| `aarch64-unknown-linux-gnu` | 103 | 103 | 0 | `c1da18d400f771d38156998b91a75a2a1e2cad3b44c13198106c22c18aea18b3` | `c161caf7dd8ea4ab45719e0cf42516fee258c111f1ec4169b0c68fb581146c91` |

All four verifier runs exited zero with no completeness gaps. Both macOS
graphs contain the ten `objc2` family HOLD rows listed above. The Android
header row is outside these four graphs; that does not clear its union HOLD.
These target-resolve results alone do not prove which optional crates are
present in a particular built wheel. No macOS gate verdict is issued.

## Python companion-file follow-up

The exact `packaging 26.2` and `pygments 2.20.0` wheel members described in
[`SOURCES.md`](SOURCES.md) are explanatory and attribution files. Their
acceptance requires the complete named grants in the same hash-bound wheel;
the member hash or wheel hash changing returns them to HOLD. The verifier's
new union rerun exited zero with no completeness gaps, and changed only these
two Python rows from HOLD to PERMISSIVE. Python HOLD is now **6**; Cargo union
HOLD remains **11**. The new inventory is
`current-python-companions/inventory.json` on s1, SHA256
`19d81c75e85511a29b3185421e12c1436f9e67072c3d8e7dcbead8ac1f74c246`.
The focused generator and SPDX tests passed: 238 tests.

The next exact-text review accepted the complete `colorama` BSD three-clause
variant and the two byte-identical `hypothesis` files containing a full MPL 2.0
body after an attribution introduction. Source hashes and the pinned SPDX
comparison are in [`SOURCES.md`](SOURCES.md). A case-sensitive BSD detector
check was corrected; the union rerun changed exactly these three Python rows
and no Cargo row. Python HOLD is now **3** (`atheris`, `numpy`,
`sortedcontainers`); both `hypothesis` rows are **WEAK-COPYLEFT**, not
PERMISSIVE. The verifier exited zero with no completeness gaps. Its inventory
is `current-python-variant/inventory.json` on s1, SHA256
`fd72fa5f302a5a89f1e9f2084980f314e5e4bcaf7af9678140ed5dacccc98583`.
The focused tests passed: 241 tests.
