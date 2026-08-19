# Item-bank transition replay callback safety

## Fixed

- Lifecycle transition replay now validates the exact creation-time record and evidence-reference instance state before invoking canonical serialization or fingerprint verification.
- Frozen lifecycle records mutated through Python object internals cannot shadow `_content_dict()` or evidence `to_dict()` callbacks to execute caller code while acquiring transition authority.
- This changes provenance/integrity validation only; calibration, fit, DIF, item-information, linking, exposure, drift, uncertainty, and other production psychometric arithmetic remain Rust-owned and unchanged.
