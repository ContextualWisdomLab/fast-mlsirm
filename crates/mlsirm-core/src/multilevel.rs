//! Validated public boundary for sparse multiple-membership contextual effects.
//!
//! The private kernel retains deterministic weighted summation and worker
//! partitioning. This boundary rejects duplicate context indices in each CSR
//! row before any numerical work, so equal-key ordering can never affect a
//! public result.
//!
//! Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
//! multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
//! 103-124. <https://doi.org/10.1177/1471082X0100100202>

use std::collections::HashSet;

#[path = "multilevel_kernel.rs"]
mod kernel;

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

/// Compute each observation's weighted contextual random-effect contribution.
///
/// The sparse CSR rows may arrive in any edge order, but each context index
/// must occur at most once in a row. All remaining shape, range, weight, and
/// worker-count validation is delegated to the private deterministic kernel.
pub fn weighted_contextual_effect(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    effects: &[f64],
    worker_count: usize,
) -> Result<Vec<f64>, String> {
    validate_unique_context_indices_per_row(row_offsets, context_indices)?;
    kernel::weighted_contextual_effect(
        row_offsets,
        context_indices,
        weights,
        effects,
        worker_count,
    )
}
