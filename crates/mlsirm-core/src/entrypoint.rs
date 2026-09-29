//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping newer implementations in
//! focused module boundaries avoids adding more several-thousand-line sections
//! to that file and gives CPU/GPU or domain-neutral numerical contracts stable
//! public homes.

include!("lib.rs");

pub mod interaction_map;
pub mod one_way_random_intercept;
pub mod rotation;
pub mod sampling_design;
