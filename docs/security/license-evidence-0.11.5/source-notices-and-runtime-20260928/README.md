# Verified integration checkpoint

Central #2468 merged e45f1b144aef900d734ff4c900f9e0010fd5a32d, whole tree 3728fb0df136a60bd19e81a349c6fe1f78af891b exactly matches the tested integration. Sidecar contracts: 31 local and 31 CI-mode tests passed. Actual guest-02 probe confirmed the same selected executable reported 3.12.3 with inherited libraries and 3.12.14 with the patched prefix. Old immutable Noema runs do not consume this fix; no hosted model-review PASS claimed.

Source notices: official archives and their ten license members independently authenticated; deployed helper positive and negative controls passed. Combined #2232/#2233 tree db2e13b7df4abaac6635f48eff8661ae6548903c passed 277 focused tests and actual offline 0.11.4 sdist validation of all 219 notices. See source receipts and exact-head review comments.

NumPy C/D both completed. Their OpenBLAS libraries match, but raw and repaired wheel hashes differ. Nine member differences include __config__.py, seven ELF files and RECORD. Build paths differ; remaining ELF bytes are not yet fully explained. This is not compilation reproducibility acceptance. Preserve original outputs; investigate before stable-path repeats.

Release helper currently pinned to 4b0c6b75 does not include #2468. Adopt the new runtime fix through the immutable helper/callee/caller chain before claiming release consumption. Original Cargo 11/Python 3 HOLD and actual published twelve-wheel verification remain unresolved. No tags or publication performed.

ELF follow-up: independent ELF64 section parsing located the remaining non-build-id differences in `.rodata`: C embeds `../../../tmp/pip-req-build-3bjizj_9/` and D embeds `../../../tmp/pip-req-build-kc2st3le/`, in two multiarray source locations and four Cython pxd locations. All other residual differences are in `.note.gnu.build-id`. The existing recipe runs pip wheel on the tarball, causing randomized extraction paths. A fresh repeat must extract the immutable tarball into the same container source path and build that directory, with identical backend/OpenBLAS paths. No existing wheel bytes were changed.

Fast #2233 merged ad1a0d204a83c7678f89e3ef2518ac2a5244ae3e; the complete merged tree is db2e13b7df4abaac6635f48eff8661ae6548903c, exactly the independently tested integration tree.
