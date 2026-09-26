# Non-Linux Cargo license review — 2026-09-26

Status: **partial**. Coordinator-style technical review delegated by the maintainer; not legal review or a release gate verdict.

Release-state check on 2026-09-26: the [PyPI project JSON](https://pypi.org/pypi/fast-mlsirm/json)
listed 0.11.4 as latest and no files for 0.11.5; the
`/pypi/fast-mlsirm/0.11.5/json` endpoint returned HTTP 404. This review
therefore concerns the prepared 0.11.5 build inputs and locally built wheel,
not a published 0.11.5 PyPI artifact matrix. A release-wide verdict requires
the exact built artifacts for every intended target.

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
The [release workflow matrix](../../../../.github/workflows/publish-pypi.yml)
plans CPython 3.12, 3.13, and 3.14 wheels for Linux x86_64, Linux aarch64,
macOS universal2, and Windows x86_64 (12 wheels). Universal2 covers both
macOS target triples below; the previously verified Linux x86_64 graph has
103 rows and HOLD 0.

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

The complete [objc2 source snapshot at the archive publication commit](https://codeload.github.com/madsmtm/objc2/tar.gz/8852b424193ca41602281b3d7540d7c8ed51e49a)
has SHA256 `e26253acf0639ce1986f79690fbc3ae9fa29b082ddc1e8661b9b53ac8d4925b3`.
Its only license-named document is `LICENSE.md`, whose SHA256
`7f976f7e9cb2d87df7230606feb932c3f21ac0e664045a775b600046ff850c54`
matches the previously fetched raw file. That document links to standard
MIT/Zlib/Apache terms and itself raises an Apple SDK licensing question; it
does not supply the complete grants. The `copying.rs` paths in the snapshot
are Rust source or tests, not additional license documents. This confirms the
ten macOS target HOLD rows cannot be cleared by an overlooked license file in
that pinned source tree.

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

The three Python HOLD rows still need separate evidence:

| Package | Exact artifact finding | Why HOLD remains |
| --- | --- | --- |
| `atheris 3.1.0` | Three CPython wheels contain `asan_with_fuzzer.so`, `ubsan_with_fuzzer.so`, `ubsan_cxx_with_fuzzer.so`, and `libclang_rt.fuzzer_no_main.a`. | Their native files have no bound license notice stanza. The wheel's Apache license covers its own candidate file, but does not establish these components' grants. |
| `numpy 2.5.2` | The pinned wheel (SHA256 `3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8`) still has 11 unverified license candidate files. | Several files combine component grants, and the main notice includes GPL/LGPL component text. The external-runtime decision does not waive all-candidate verification. |
| `sortedcontainers 2.4.0` | Its wheel `LICENSE` (SHA256 `1db7cae7fce6452e2e608e401a0f953e0133e4c2d75db69fb8ae851d2086f5b6`) is a short Apache header and URL. | The complete Apache 2.0 terms are absent from the hash-bound wheel candidate; a URL alone is not a verified full grant under the current rule. |

The [Atheris 3.1.0 PyPI release](https://pypi.org/project/atheris/3.1.0/)
has only three wheels and no source distribution. Their SHA256 digests are
`ec5e11f21a4c197fe91f7aea2b2de88e623c73a21fc07b105ac6329a1588457b`
(cp312), `f8a9f51ce8369026e8eb7b7174835e8c4c85a1a6db5d9add36c15100779d2a39`
(cp313), and `315a0b5c819852b1ffe1ca72efc389c7724881f2c33e4aacb8c6bcec49bd5011`
(cp314), matching the inventory inputs. All three contain the same four
native-file hashes. [Upstream build code](https://github.com/google/atheris/blob/master/setup.py)
shows that Atheris copies libFuzzer and combines it with ASan/UBSan, but that
moving branch does not bind these exact wheel binaries to an LLVM version,
source commit, or license notice. No native exception was added.
The native SHA256 digests are `3d5fbe5d97101964713e476f85c21393d769a7ad3355538eba1e36590c68bed7`
(`asan_with_fuzzer.so`), `60d06f6748c007c46c772b0abee959053973ee4b607a88f9f3db3562b2bbecaf`
(`libclang_rt.fuzzer_no_main.a`), `d77a11a9024b34aa37c86b41f78be3ce2d7ce59067208c1fb662cd216c50eb1c`
(`ubsan_cxx_with_fuzzer.so`), and `ffbe77bee5e88de99ad4b52622dcee38c9ea56a563ad63b1d4eeba35e6ae5d93`
(`ubsan_with_fuzzer.so`).

NumPy's main wheel `LICENSE.txt` is SHA256
`4860083caa0de2ac3292ca98bd074bd8f45d8b32624e37b1e70a240bff61e488`.
It embeds full BSD, GCC runtime exception, and GPL 3 text, then names
`libquadmath` as LGPL 2.1 or later with a short notice and a link instead of
the complete LGPL text. Several other candidate files contain multiple
component grants. Because the main candidate alone remains incomplete under
the current all-candidate rule, accepting individual NumPy component hashes
would not clear this row; it stays HOLD pending a complete, bound notice set.

The [sortedcontainers 2.4.0 source distribution](https://pypi.org/project/sortedcontainers/2.4.0/#files)
has SHA256 `25caa5a06cc30b6b83d11423433f65d1f9d76c4c6a0c90e3379eaa43b9bfdb88`.
Its `LICENSE` bytes hash to the same
`1db7cae7fce6452e2e608e401a0f953e0133e4c2d75db69fb8ae851d2086f5b6`
as the wheel's short Apache header. The release source therefore supplies no
missing full text; this row stays HOLD.
