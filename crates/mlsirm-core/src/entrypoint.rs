//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping the rotation implementation
//! in its own module boundary avoids adding another several-thousand-line
//! section to that file and gives future CPU/GPU backends a stable public home.

include!("lib.rs");

pub mod interaction_map;
pub mod rotation;
