# Exact whole-text support for Cargo evidence

Combined base: dc41563f523f977a380974c0644040c7c4eab657. Scope: four complete
Apache-2.0/Zlib texts recognized by whole-text SHA256 after ASCII layout
whitespace normalization only. Package names, SPDX labels, filenames, and
partial text do not substitute for recognition. Every candidate file is still
checked. Existing MIT recognition and SPDX election/AND logic are unchanged.

`tests/fixtures/license_inventory_reviewed_texts.json` preserves the complete
UTF-8 texts with artifact SHA256, exact archive member, raw SHA256, normalized
SHA256 and identifier. The existing full Apache and foldhash Zlib evidence comes
from central d4d80fda's reviewed `tests/fixtures/release_license_texts/texts.json`
and `provenance.json`. These recognize text, not the entire originating package.
The allocator-api2 and glow variants are read directly from existing s1 cache
archives under `/data/orca/workspaces/fmls-license-evidence/cargo-home/registry/cache/`.
Their raw member hashes match the preserved 63aa inventory summary.

Source-check procedure: this is an artifact-text comparison, not a paper or
legal opinion. Archive/member locators replace PDF page locators; no OCR is
involved. Complete normalized-token comparison finds the allocator-api2 Apache
variant differs only by omission of the application APPENDIX, with the full
terms retained. The glow Zlib variant differs only by omission of foldhash's
specific copyright line; the complete permission/restriction/disclaimer body
remains. These exact variants receive their own digests. No generic appendix or
copyright stripping is implemented. Any other header or additional condition
remains unknown. Original copyright-bearing variants remain distinct evidence.

This can address the currently observed canonical-validator gap without
skipping unselected files. The preserved read-only comparison groups 33
Apache-only holds into five existing full-text matches and 28 exact no-appendix
variants. glow additionally needs the exact headerless Zlib variant. No corpus
rerun confirms updated row classifications. The four LLVM/COPYRIGHT cases,
individual-file exceptions, unverified selected MIT (including r-efi), NumPy
vendored-native evidence, and complete release scope remain HOLD.

Tests exercise exact bytes and hashes, ASCII line endings, one-word mutations,
extra conditions, added prefix/suffix, and the real synthetic crate inventory
consumer with an unverified COPYRIGHT, missing selected MIT, extra restriction,
or LGPL material. Existing election and AND regressions remain in the same
focused file. The NumPy repair and exact-text support are combined in commit
7354cf2ca3f694355d94d502d8487e61a41dba57.

## 66 copyright-header/appendix-only Cargo variants (2026-09-26)

Sixty-six more whole texts (MIT 55, Apache-2.0 8, Zlib 3) from the 132
binding-graph HOLD rows of s1 inventory eeb2ed00 are recognized by the same
exact normalized-SHA256 lookup. Each was extracted from its lock-checksum-matched
`.crate` on s1 and compared with a local template (MIT body from this module,
`/usr/share/common-licenses/Apache-2.0`, foldhash Zlib). The only differences are
leading title/copyright lines or an omitted or filled Apache APPENDIX. Per-hash
quotes are in s1 `text-review-20260926/copyright-only-diffs.md` (sha256
3a44b440bf93462b102f97b77bf41836bd9ab6ab05015faa64fd085038a9f8a4). Matching logic
is unchanged. The 10 DEVIATES texts, 11 NO_LOCAL_TEMPLATE texts, and crates
with no license file remain HOLD.

## Documented exception: libm 0.2.16 (2026-09-26)

Coordinator verdict (user-delegated, 2026-09-26), not legal review. `DOCUMENTED_EXCEPTIONS` in
`tools/license_inventory.py` accepts libm 0.2.16 `LICENSE.txt` (sha256
3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7) as notice-preserving
permissive MIT, only inside the lock-matched `.crate` sha256
b6d2cec3eae94f9f509c767b45932f1ada8350c4bdb85af2fcab4a3c14807981. libm is linked into the Linux
cp312 `_core` via naga/wgpu and num-traits. Its sources carry Sun "freely granted, provided that
this notice is preserved" (50 files), BSD-2-Clause (David Schultz, 2 files) and MIT (core-math,
8 files) notices. Evidence: s1 `libm-notice-20260926/` (PROPOSAL.md sha256 bdeac27d...).

**Release gate (recorded, not yet enforced):** the wheel's third-party notices must include libm
`LICENSE.txt` and the Sun, BSD-2-Clause (David Schultz) and core-math MIT source notices before
any release that ships the Rust core.

## Documented exception: cfg_aliases 0.2.2 NOTICES.md (2026-09-26)

Coordinator verdict (user-delegated, 2026-09-26), not legal review. `DOCUMENTED_EXCEPTIONS`
accepts cfg_aliases 0.2.2 `NOTICES.md` (sha256
1e2b7ade3fb228130408b9990cae6a7618eb314c75aa0b164bfe485d9d9756ee) as a third-party MIT
attribution notice, only inside the lock-matched `.crate` sha256
f079e83a288787bcd14a6aea84cee5c87a67c5a3e660c30f557a3d24761b3527. Its embedded MIT text matches the
SPDX v3.29.0 MIT matching template (curly-quote equivalence). **Recorded gap:** there is no
copyright-holder line for tectonic_cfg_support.

**Release gate (recorded, not yet enforced):** the wheel NOTICE must also include cfg_aliases
`NOTICES.md`.

## Python side (2026-09-26)

Coordinator verdict (user-delegated, 2026-09-26), not legal review. The same criterion as Cargo applies
(SPDX matching template with every difference inside a `<<var>>`/`<<beginOptional>>` span, and the
copyright span holding only title/copyright lines). Accepted: build, iniconfig, pluggy, pytest,
pyproject-hooks, maturin (MIT) and pygments, packaging LICENSE.BSD (BSD-2-Clause).

numpy 2.5.2 is recorded in `EXTERNAL_RUNTIME_DEPENDENCIES` as an external runtime dependency that
fast-mlsirm never bundles, pinned to the official wheel sha256 3cdec01f...: its vendored native
libraries (libgfortran, libquadmath, OpenBLAS) are reported as notes, not HOLD. Its own license
candidate files must still verify.

atheris 3.1.0 remains HOLD as CI-only (fuzz extra; never shipped).

Zip directory entries (for example `numpy-2.5.2.dist-info/licenses/`) are no longer treated as
empty license candidate files.
