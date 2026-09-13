//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping independently owned domain
//! modules outside that file avoids adding more several-thousand-line sections
//! and gives downstream contexts stable public boundaries.

include!("lib.rs");

pub mod governed_rater_contracts;
pub mod interaction_map;
pub mod rotation;
pub mod sampling_design;
