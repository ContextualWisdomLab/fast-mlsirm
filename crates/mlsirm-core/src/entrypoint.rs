//! Crate entrypoint that preserves the established core while adding modular domains.
//!
//! The historical core remains in `lib.rs`. New reusable numerical responsibilities
//! live in focused modules so product-specific temporal or transport policy does not
//! leak into the psychometric kernel.

include!("lib.rs");

pub mod interaction_map;
pub mod rotation;
pub mod sampling_design;
pub mod standardisation;
