//! Validated public boundary for sparse multiple-membership contextual effects.
//!
//! The private kernel retains deterministic weighted summation and worker
//! partitioning. This boundary rejects duplicate context indices in each CSR
//! row and non-finite referenced contextual effects before numerical work, then
//! rejects any non-finite weighted output before it can escape the public
//! boundary. Finiteness validation follows only referenced context indices so
//! unused effect-table capacity cannot expand work independently of the sparse
//! design. Equal-key ordering therefore cannot affect a public result and finite
//! input overflow cannot silently become an infinite contextual contribution.
//!
//! [`estimate_crossed_person_effects`] is the MAP estimator of the same `u_h`
//! vector for crossed and weighted multiple-membership designs (Fox & Glas,
//! 2001; Browne, Goldstein, & Rasbash, 2001). It does not rewrite the
//! longitudinal OLS / AR utilities reserved for a separate slice.
//!
//! Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
//! multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
//! 103-124. <https://doi.org/10.1177/1471082X0100100202>
//!
//! Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
//! IRT model. *Psychometrika, 66*, 271-288.
//! <https://doi.org/10.1007/BF02294839>

use std::collections::HashSet;

#[path = "multilevel_estimator.rs"]
mod estimator;
#[path = "multilevel_kernel.rs"]
mod kernel;

pub use estimator::{
    estimate_crossed_person_effects, CrossedPersonEffectConfig, CrossedPersonEffectEstimate,
    MAX_CROSSED_EFFECTS, MAX_CROSSED_ITER,
};

/// Maximum contextual-membership edges accepted at the public Rust boundary.
pub const MAX_CONTEXT_MEMBERSHIPS: usize = 100_000;

fn validate_unique_context_indices_per_row(
    row_offsets: &[usize],
    context_indices: &[usize],
) -> Result<(), String> {
    if row_offsets.first() != Some(&0)
        || row_offsets.last().copied() != Some(context_indices.len())
        || row_offsets.windows(2).any(|window| window[1] < window[0])
    {
        // Preserve the kernel's established malformed-CSR error messages.
        return Ok(());
    }

    let mut seen = HashSet::new();
    for window in row_offsets.windows(2) {
        seen.clear();
        for &index in &context_indices[window[0]..window[1]] {
            if !seen.insert(index) {
                return Err("context_indices must be unique within each row".to_string());
            }
        }
    }
    Ok(())
}

fn validate_referenced_finite_effects(
    context_indices: &[usize],
    effects: &[f64],
) -> Result<(), String> {
    for &index in context_indices {
        let effect = effects
            .get(index)
            .ok_or_else(|| "context_indices contains an index outside effects".to_string())?;
        if !effect.is_finite() {
            return Err("effects must be finite".to_string());
        }
    }
    Ok(())
}

/// Compute each observation's weighted contextual random-effect contribution.
///
/// The sparse CSR rows may arrive in any edge order, but each context index
/// must occur at most once in a row and every referenced contextual random-
/// effect value must be finite. Unreferenced effect-table entries are not
/// scanned because they cannot affect this design's output. A weighted sum that
/// overflows despite finite referenced inputs is also rejected. All remaining
/// shape, weight, and worker-count validation is delegated to the private
/// deterministic kernel.
pub fn weighted_contextual_effect(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    effects: &[f64],
    worker_count: usize,
) -> Result<Vec<f64>, String> {
    if context_indices.len() > MAX_CONTEXT_MEMBERSHIPS {
        return Err(format!(
            "context_indices exceeds the membership-edge cap of {MAX_CONTEXT_MEMBERSHIPS}"
        ));
    }
    validate_unique_context_indices_per_row(row_offsets, context_indices)?;
    validate_referenced_finite_effects(context_indices, effects)?;
    let output = kernel::weighted_contextual_effect(
        row_offsets,
        context_indices,
        weights,
        effects,
        worker_count,
    )?;
    if output.iter().any(|effect| !effect.is_finite()) {
        return Err("weighted contextual effects must be finite".to_string());
    }
    Ok(output)
}
