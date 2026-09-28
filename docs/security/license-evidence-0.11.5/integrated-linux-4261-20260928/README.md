# Current A3 source: two additional Linux x86_64 candidate wheels

Source commit: `4261e8e6e00c10ce8f1fc068db1a7dd0f24edd46` (draft PR #2177).
The owner's sparse `source.tar.gz` has SHA256
`eb98cb0272dedf2a758e3a172b48bd5b63cf98251be680ff387e79473f0c977a`.
Before building, all 553 regular archive members were compared byte for byte
with that clean checkout; none differed. `SOURCE_DATE_EPOCH=1790553578` is the
commit time. The owner workspace was read only.

The retained script builds in an isolated s1 directory,
`/data/orca/workspaces/fmls-a3-4261-linux-x86-cp313314-20260928`, with the
pinned `manylinux2014_x86_64` image
`sha256:21c37461985655aaa25ed3a923b28e6c9d4dd9e10edc0281eb50385184bddd31`,
Rust 1.97.1, maturin 1.15.0, `--offline --locked`, two CPUs and 6 GiB. It
copies the owner's CPython 3.12 Cargo target cache before the two new ABI
builds; these are fresh ABI wheel outputs, not an independent clean-cache
recompilation. No owner source directory or running container was modified.

| CPython | Wheel SHA256 |
| --- | --- |
| 3.13 | `9c357d48c317d2ebb515970698217e39635f7b0806e14b6058ff435c5b69918e` |
| 3.14 | `1660a6223c8d8ea79650f24c24154f5acf15d0ba13aed9e9e6048a37baf64877` |

For each exact wheel, the built-wheel license verifier passed with selected
`LICENSE-THIRD-PARTY` SHA256
`d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0`.
ZIP CRC, `METADATA Version: 0.11.5`, six `License-File` entries, matching
CPython ABI tags and `auditwheel` `manylinux_2_17_x86_64` checks passed.
The build, license, auditwheel and tool-install logs are retained here and
hashed in `SHA256SUMS`. The owner's separate CPython 3.12 wheel and ten
installed regression tests are recorded in `CONTINUATION-20260928.md`.

These are local Linux candidates. Installed CPython 3.13/3.14 numerical tests,
Windows/macOS candidates, published twelve-wheel hashes, target-specific
license closure and release acceptance remain unverified. The original Cargo
and Python HOLDs are unchanged.
