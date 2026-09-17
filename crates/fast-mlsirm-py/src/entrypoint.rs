//! PyO3 entrypoint preserving `_core` and registering modular extensions.
//!
//! PyO3 permits one shared library to export multiple `PyInit_*` symbols. The
//! historical module remains `_core`; bounded domain modules export additional
//! initialization symbols from the same binary. Keeping registrations together
//! prevents stacked feature branches from silently replacing one another's
//! extension-module entrypoint.

include!("lib.rs");

mod ata_bindings;
mod bifactor_bindings;
mod interaction_map_bindings;
mod multilevel_bindings;
mod rating_range_bindings;
mod rotation_bindings;
