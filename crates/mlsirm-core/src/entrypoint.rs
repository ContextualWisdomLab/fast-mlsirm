//! Crate entrypoint that preserves the established core while adding modular rotation.
//!
//! The historical core remains in `lib.rs`. Keeping the rotation implementation
//! in its own module boundary avoids adding another several-thousand-line
//! section to that file and gives future CPU/GPU backends a stable public home.

include!("lib.rs");

pub mod dif_information_criteria;
pub mod interaction_map;
pub mod normal_quantile;
pub mod rotation;
pub mod samefit_grm;
#[cfg(feature = "gpu")]
pub mod samefit_grm_gpu;
pub mod samefit_grm_joint;
pub mod samefit_strict;
pub mod sampling_design;
#[cfg(all(feature = "gpu", not(coverage)))]
pub mod gpu_mstep;
#[cfg(all(feature = "gpu", not(coverage)))]
mod gpu_grm_estep;

pub mod samefit_fifa;
#[cfg(all(feature = "gpu", not(coverage)))]
pub mod two_tier_gpu_item;
