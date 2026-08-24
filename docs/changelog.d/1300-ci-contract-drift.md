# Close CI contract drift on the toolchain pin and metadata scalar admission

## Fixed

- Pin the `grm-recovery` scheduled statistical-study job's `dtolnay/rust-toolchain` step to exact Rust `1.97.1`, closing a gap where it silently floated to the default stable channel while every sibling verification lane stayed pinned.
- Align `test_metadata_rejects_string_subclasses_before_callbacks` with the metadata scalar admission boundary's actual, intentional behavior: caller-defined `str` subclasses are safely normalized through the inert `str.__str__` descriptor (matching the established `int`/`float` subclass handling in the same function) without invoking any subclass-defined method, rather than being rejected outright.
