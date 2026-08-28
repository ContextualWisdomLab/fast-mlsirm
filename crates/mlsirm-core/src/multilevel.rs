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
    CrossedPersonEffectConfig, CrossedPersonEffectEstimate, MAX_CROSSED_EFFECTS,
    MAX_CROSSED_ITER,
};

/// Maximum contextual-membership edges accepted at the public Rust boundary.
pub const MAX_CONTEXT_MEMBERSHIPS: usize = 100_000;
/// Maximum CSR row-pointer entries accepted at the public Rust boundary.
pub const MAX_CONTEXT_ROW_OFFSETS: usize = MAX_CONTEXT_MEMBERSHIPS + 1;

fn preflight_crossed_estimator_controls(
    y: &[f64],
    classification_offsets: &[usize],
    n_persons: usize,
    n_items: usize,
    n_effects: usize,
    config: &CrossedPersonEffectConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 || n_effects < 1 {
        return Err("n_persons, n_items, and n_effects must be at least one".to_string());
    }
    if n_effects > MAX_CROSSED_EFFECTS {
        return Err(format!(
            "n_effects exceeds the dense Newton cap of {MAX_CROSSED_EFFECTS}"
        ));
    }
    if classification_offsets.len() > n_effects + 1 {
        return Err("classification_offsets exceeds n_effects + 1".to_string());
    }
    let expected = crate::checked_mul_usize(n_persons, n_items, "response matrix is too large")?;
    if expected > estimator::MAX_CROSSED_RESPONSE_CELLS {
        return Err(format!(
            "crossed response matrix exceeds the logical-cell cap of {}",
            estimator::MAX_CROSSED_RESPONSE_CELLS
        ));
    }
    if y.len() != expected {
        return Err("y must have length n_persons * n_items".to_string());
    }
    if !config.prior_precision.is_finite() || config.prior_precision <= 0.0 {
        return Err("prior_precision must be finite and strictly positive".to_string());
    }
    if !(1..=MAX_CROSSED_ITER).contains(&config.max_iter) {
        return Err(format!("max_iter must be in 1..={MAX_CROSSED_ITER}"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err("tol must be finite and strictly positive".to_string());
    }
    if !(1..=estimator::MAX_CROSSED_WORKERS).contains(&config.worker_count) {
        return Err(format!(
            "worker_count must be in 1..={}",
            estimator::MAX_CROSSED_WORKERS
        ));
    }
    Ok(())
}

/// Estimate crossed / multiple-membership person effects after bounded control preflight.
///
/// Cheap dimension, response-work, response-length, and execution-control validation
/// runs before any response-value traversal. The private estimator repeats those
/// invariants as defense in depth and owns all likelihood, score/information,
/// Newton, centering, and CPU/GPU numerical work.
#[allow(clippy::too_many_arguments)]
pub fn estimate_crossed_person_effects(
    y: &[f64],
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    person_offsets: &[f64],
    classification_offsets: &[usize],
    n_persons: usize,
    n_items: usize,
    n_effects: usize,
    config: CrossedPersonEffectConfig,
) -> Result<CrossedPersonEffectEstimate, String> {
    preflight_crossed_estimator_controls(
        y,
        classification_offsets,
        n_persons,
        n_items,
        n_effects,
        &config,
    )?;
    estimator::estimate_crossed_person_effects(
        y,
        row_offsets,
        context_indices,
        weights,
        item_slopes,
        item_intercepts,
        person_offsets,
        classification_offsets,
        n_persons,
        n_items,
        n_effects,
        config,
    )
}

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
    if row_offsets.len() > MAX_CONTEXT_ROW_OFFSETS {
        return Err(format!(
            "row_offsets exceeds the CSR row-pointer cap of {MAX_CONTEXT_ROW_OFFSETS}"
        ));
    }
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