//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping newer implementations in
//! explicit module boundaries avoids adding another several-thousand-line
//! section to that file and gives reusable numerical contracts stable public homes.

include!("lib.rs");

pub mod covariance_standardization;
pub mod interaction_map;
pub mod rotation;
pub mod sampling_design;
