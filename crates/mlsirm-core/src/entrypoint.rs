//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping the rotation implementation
//! in its own module boundary avoids adding another several-thousand-line
//! section to that file and gives future CPU/GPU backends a stable public home.

include!("lib.rs");

pub mod rotation;

// The criterion-neutral selector is kept in its own source file. Its implementation
// expects its parent module to provide the core rotation types, so crate-private
// aliases make that contract explicit without widening the package-root API.
pub(crate) use rotation::{
    rotate_factor_loadings, RotationConfig, RotationCriterion, RotationSolution,
};
#[path = "rotation/selector.rs"]
pub mod rotation_selection;
