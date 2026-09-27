# Verified integration checkpoint

Central #2468 merged e45f1b144aef900d734ff4c900f9e0010fd5a32d, whole tree 3728fb0df136a60bd19e81a349c6fe1f78af891b exactly matches the tested integration. Sidecar contracts: 31 local and 31 CI-mode tests passed. Actual guest-02 probe confirmed the same selected executable reported 3.12.3 with inherited libraries and 3.12.14 with the patched prefix. Old immutable Noema runs do not consume this fix; no hosted model-review PASS claimed.

Source notices: official archives and their ten license members independently authenticated; deployed helper positive and negative controls passed. Combined #2232/#2233 tree db2e13b7df4abaac6635f48eff8661ae6548903c passed 277 focused tests and actual offline 0.11.4 sdist validation of all 219 notices. See source receipts and exact-head review comments.

NumPy C/D both completed. Their OpenBLAS libraries match, but raw and repaired wheel hashes differ. Nine member differences include __config__.py, seven ELF files and RECORD. Build paths differ; remaining ELF bytes are not yet fully explained. This is not compilation reproducibility acceptance. Preserve original outputs; investigate before stable-path repeats.

Release helper currently pinned to 4b0c6b75 does not include #2468. Adopt the new runtime fix through the immutable helper/callee/caller chain before claiming release consumption. Original Cargo 11/Python 3 HOLD and actual published twelve-wheel verification remain unresolved. No tags or publication performed.
