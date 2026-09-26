# A3 license gate verdict — Linux x86_64 cp312, fast-mlsirm 0.11.5

Status: **provisional for the examined Linux cp312 candidate** (coordinator verdict, user-delegated, 2026-09-26; not legal review). Independent technical review found the cp312 code verdict non-blocking; hosted checks, GitHub non-author approval, and release remain pending.
Decision records: coordinator messages msg_60a758c73aad, msg_58e4fad25e39, msg_ee85fb3cb33f, msg_13c9108bac7d, msg_b5aa37b1b0c7, msg_411f3daec1fa.

## Scope
- Local integrated source candidate: merge commit `f49924c87f7d6f9004ea590c48cc509dd9f31c5f` with parents #2157 `58b7b23f7a154c8391c130ebc3aab2dde50bb84b` and #2170 `80c240d2bc5b24974bd91a25815e9a1b3d574aa8`. The `uv.lock` conflict was resolved with #2157's NumPy 2.5.2; both hash-locked requirements files also retain 2.5.2. `uv lock --check --offline` passed.
- Candidate artifact checked: `fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  SHA256 `d8ec1d497763abd943dfc5ba0defa93a67f141b8bab9adf04a02c8d09a9d43bb` (built on s1 from the **#2157 first parent**, source archive SHA256 `f8a15e7513f63b765773d5de5fa7c15bc205b3f8e6f24a1cb30097cc923094bb`). The integrated merge commit itself has **not** been rebuilt as a wheel; its six notice files, Cargo locks, Python locks, and inventory tool match the examined inputs byte-for-byte. `auditwheel` reports `manylinux_2_17_x86_64`; the installed #2157 wheel passed 10 regression tests.
- Cargo binding graph for `x86_64-unknown-linux-gnu`: 103 rows, **HOLD 0 after the actual-wheel correction below**. Non-Linux targets and 12-wheel expansion are outside this verdict. The #2170 second parent alone has 0.11.4 manifests and cannot establish the 0.11.5 result.

## Actual-wheel correction and binding

The previous generator applied all six wheel `License-File` entries to each own crate as candidate license texts. With the real wheel and a measured 64-character SHA256, that gave Linux **101 PERMISSIVE, 2 HOLD**: both own crates rejected five supplemental files as noncanonical, including two unrecognized notices. The corrected common path retains all six files as evidence, tests only the project's `LICENSE` as the own-crate MIT grant, and records the other five as distribution or named third-party attribution. It does not mark those five as canonically verified license texts. An unknown notice owner remains HOLD.

The integrated candidate's inventory run uses the #2157 wheel above, the integrated 0.11.5 Cargo and Python locks, `--own-crate-wheel-source-root` pointing to the byte-identical #2157 source files, and the Linux target graph. Every declared wheel file must match its source bytes. `LICENSE-THIRD-PARTY` must match the tracked snapshot SHA256 `e6d920bd6bdd392cc902a5a54fe365f48fdadb1b3762e857fddee1e4dea6ca26`, its source inventory SHA256 `621e4ca2e51b460c832438c1ec04cb679ca44c2dddad396d08103c2e99590d28`, and all 101 third-party rows in the target graph. The two own crate rows are version 0.11.5, PERMISSIVE, with no HOLD reasons; all 103 Linux rows are PERMISSIVE and completeness gaps are empty. Current integrated-run inventory SHA256: `0aefa45bbe77f2ca7d25302f918efe049091a16c6c4ca72372ca71870891b3a3` at s1 `runs/fmls-a3-integrated-f49924c8-20260926/inventory.json`. This is local cp312 evidence, not a hosted gate result.

The actual wheel's six `METADATA License-File` members have the following SHA256 values. Each corresponding top-level file in the exact #2157 source has the **same** SHA256. The real-wheel regression pins all six expected hashes, including the five supplemental files.

| Source / wheel license member | Expected SHA256 |
| --- | --- |
| `LICENSE` | `08f1fd81fb120bc468b69dc3e58ea0dc23c216305c766e45e107f56c76559e3f` |
| `LICENSE-THIRD-PARTY` | `d46f307a2e8a49e2d638ee4e0b768c6c908c7786cb9106e74948561bf8cf0af0` |
| `NOTICE` | `7192b2614bfee6e95283ef9db9f5fe41c2acb579f5cd30e6482445e42fca0ae5` |
| `NOTICE-cfg_aliases-0.2.2-NOTICES.md` | `1e2b7ade3fb228130408b9990cae6a7618eb314c75aa0b164bfe485d9d9756ee` |
| `NOTICE-libm-0.2.16-LICENSE.txt` | `3823dda7cf046602f4b4e77ec8e227863dc4736037cc85bb33d9f19febe16bb7` |
| `NOTICE-libm-0.2.16-source-notices.txt` | `9e949a13f66c0f9b60b73b54e8ab2940ccff92d704c46c53103b1028e2cc75ba` |

Run the real-wheel hash regression explicitly with `python tests/test_license_inventory_generator.py WHEEL EXACT_SOURCE_ROOT`; the default pytest collection needs no wheel and has no skip. On the integrated merge tree the focused SPDX/inventory set passed 236/236; the explicit actual-wheel check passed, and a copy with a changed source `NOTICE` was rejected. The source archive identifies the #2157 first parent. The merged source tree exists at `f49924c8`, but no wheel built from that merge commit has been examined.

**Required A3 release evidence:** Before accepting this Linux cp312 candidate, run the explicit check above with a SHA-pinned wheel and its exact source archive. Record the command, both input SHA256 values, exit code, and six-file result alongside the inventory. A default pytest pass does not satisfy this artifact check. With either path omitted, the command exits 1 (`usage: test_license_inventory_generator.py WHEEL EXACT_SOURCE_ROOT`); with the cited first-parent wheel and source it prints `actual A3 wheel license roles and six source hashes verified` and exits 0. A wheel built from the integrated commit, hosted gates, and GitHub approval remain separate release conditions.

## Integrated candidate replay receipt

The exact command is tracked as [`A3-integrated-replay-f49924c8.sh`](A3-integrated-replay-f49924c8.sh) and was run on s1 as `/data/orca/workspaces/fmls-license-evidence/runs/fmls-a3-integrated-f49924c8-20260926/run.sh` (both SHA256 `6f9e915883e293fd62c9b7b987430b2273299fe697c9ff9564b10b806f67a603`). It invokes the candidate's byte-identical `tools/license_inventory.py` (SHA256 `38d969ca3dff38f955658ad40e9973bde33cb17357055d77908397a42b76fb0e`) with the #2157 wheel SHA above, Linux target metadata, the integrated candidate locks, and `--own-crate-wheel-source-root` set to the extracted #2157 source. The command exited 0; `stdout.log`, `stderr.log`, `exit_code`, and the inventory are beside `run.sh`. Key input SHA256 values:

| Input | SHA256 |
| --- | --- |
| workspace Cargo metadata | `ff4c475a22050a22749c7bd73a52ec11986673fdacfd458ddde10eb63a844cb5` |
| binding Cargo metadata | `6af5b81e471dc6c5bd4ac67617fb351e04e01b5f194d1f541dafa5f660bfbf3e` |
| Linux target Cargo metadata | `65ae955d5ec9140f1e420a8c5386a55a040c84e93d280e9dc6201bdbe938bb55` |
| wheel SBOM (byte-identical to the #2157 wheel's `dist-info/sboms/fast-mlsirm-py.cyclonedx.json`) | `00f486568b3b9ef17f81e9990ddf445de39ae3f11bae75fafdb45bddba53dcce` |
| workspace / binding `Cargo.lock` | `0601a7b4b21d3230af5032449e7eb97029c543429e0824bbf6bbed7999d61b1a` / `528414b582b256a6ab8ac68ceceb134e8ddb719774513ab8cc3a0e4eb22a177d` |
| `uv.lock` / CI / package requirements | `928cc25a0bbe7572757e68c4df42328167adb10b9ddb52ada5f8a1924a836321` / `ed59f7311c732db7d6095a2e4939c2d328ebf647794b31b89e67db29b6351975` / `f3e117fca202d27b2fe086eb1fc41644d4fef48bd5379d7a1578dca00717836d` |
| Python scope declaration | `51fb36a31525f3976cc6f3ccbed8a4d578448dd66af5855fb82e9a63e0aeea7c` |
| 101-crate source snapshot | `e6d920bd6bdd392cc902a5a54fe365f48fdadb1b3762e857fddee1e4dea6ca26` |
| executable two-entry upstream evidence | `2f505b81aeadbeec0cef3715e19bcad51ef0671447d6ec90c236f99821ae3597` |
| inventory output | `0aefa45bbe77f2ca7d25302f918efe049091a16c6c4ca72372ca71870891b3a3` |

The two-entry upstream evidence is s1 `runs/fmls-cfg-aliases-0115-20260926/upstream-evidence.json`. It binds `profiling@1.0.18` to `aclysma/profiling` commit `8271551172eb6fa4cba47369aedd93790c623df9` (`LICENSE-APACHE` SHA256 `10d30a673cd5e9349bdc02aeb48f14b3386d27d0da32df8f0a555d4aa16aa551`; `LICENSE-MIT` SHA256 `c8167fdeeed46d3f244d3f85c5bf998ce889343691c32be2c61a8bc4b5c08333`) and `spirv@0.4.0+sdk-1.4.341.0` to `gfx-rs/rspirv` commit `8afc3d0ac8e158128cd1410bb2e4b4c26ab11bb4` (`LICENSE` SHA256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`). All three local source text files were rehashed and matched the evidence. This input supplies the published-commit license text for two Linux rows. The tracked 16-entry `cargo-upstream-license-evidence.json` is a review record, **not** the executable input: its entries lack `local_path`, and passing it to the runner fails with `KeyError: local_path`. The runnable file and referenced source texts are retained on s1; reproducing the complete inventory elsewhere requires copying those verified inputs. The 101 `.crate` source hashes are checked against the tracked snapshot and Cargo lock within the inventory.

In the replayed output, the Linux Rust graph is 103 PERMISSIVE / HOLD 0 and has no completeness gaps. This is distinct from the Python inventory: separately installed NumPy 2.5.2 is a runtime dependency outside the published fast-mlsirm wheel and its Python row is **HOLD** because bundled native license texts in the official NumPy artifact were not fully verified. Neither the Rust verdict nor the Fortran-free research installation clears that Python HOLD.

## Acceptance bases (Linux graph, 103 rows)
canonical verifier text 21; byte-identical condition clauses (reviewed hash) 68; SPDX matching template 4;
upstream-vcs text at the published commit 2; own crates via published-wheel LICENSE 2; reviewed pointer notices 4;
hash-pinned documented exceptions 2 (libm 0.2.16 notice-preserving permissive; cfg_aliases 0.2.2 third-party MIT attribution).
An independent strict SPDX-template re-judgment found verifier diff 0; its 4 disagreements (cfg_aliases, libm,
matrixmultiply, ndarray) are all accepted by explicit verdicts above, not by the template criterion.

## Attribution in the candidate wheel
`dist-info/licenses/`: LICENSE, LICENSE-THIRD-PARTY (101 crates, sha256 `d46f307a…`), NOTICE, libm LICENSE and complete source notices (sha256 `9e949a13f66c0f9b60b73b54e8ab2940ccff92d704c46c53103b1028e2cc75ba`), cfg_aliases NOTICES.md. The libm notice bytes match the exact-head source.

## Outside this gate
- NumPy 2.5.2 is an unbundled external runtime dependency. Its separate Python HOLD is recorded above; research consumption used a Fortran-free NumPy build.
- Python non-shipping tooling (atheris, colorama, hypothesis x2, numpy text files, packaging, pygments, sortedcontainers): tracked as a separate license-hygiene item; none is GPL/LGPL/AGPL (strongest: MPL-2.0).
