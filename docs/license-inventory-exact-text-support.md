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
