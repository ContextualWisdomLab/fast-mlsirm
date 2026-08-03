//! PyO3 entrypoint preserving the established `_core` module and adding rotation.
//!
//! PyO3 permits a single shared library to export more than one `PyInit_*`
//! symbol. The historical module remains `_core`; `rotation_bindings` exports
//! `_rotation_core` from the same binary so the large, stable binding file does
//! not need an unrelated mechanical rewrite.

include!("lib.rs");

mod rotation_bindings;
