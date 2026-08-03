//! PyO3 entrypoint preserving `_core` and registering modular extensions.
//!
//! PyO3 permits one shared library to export multiple `PyInit_*` symbols. The
//! historical module remains `_core`; the bifactor and rotation bindings export
//! `_bifactor_core` and `_rotation_core` from the same binary. Keeping the
//! registrations together prevents stacked feature branches from silently
//! replacing one another's extension-module entrypoint.

include!("lib.rs");

mod bifactor_bindings;
mod rotation_bindings;
