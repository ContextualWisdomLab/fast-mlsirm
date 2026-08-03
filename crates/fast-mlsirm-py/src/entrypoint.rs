//! PyO3 entrypoint preserving `_core` and adding bifactor scoreability.
//!
//! The historical bindings remain in `lib.rs`. The secondary
//! `PyInit__bifactor_core` symbol is exported from the same shared library so
//! the scoreability surface stays modular and the established binding file does
//! not require a mechanical rewrite.

include!("lib.rs");

mod bifactor_bindings;
