//! Public fit-statistics boundary.
//!
//! The numerical implementations live in `fitstats_impl.rs`. This module keeps
//! their stable public surface while enforcing the versioned local-dependence
//! resource contract before the canonical Rust LD kernel can allocate its ICC
//! grid or pair vectors. Python mirrors the same ceilings as an earlier caller
//! boundary; the Rust contract remains authoritative for every direct caller.

#![allow(hidden_glob_reexports)]

#[path = "fitstats_impl.rs"]
mod implementation;

pub use implementation::*;

use crate::nodes::{XiRule, MAX_XI_LATENT_DIM, MAX_XI_POINTS};
use crate::scoring::{validate_bank, ItemBank, PriorSpec};

/// Version of the native local-dependence resource/work contract.
pub const LD_RESOURCE_CONTRACT_VERSION: &str = "ld-resource-v1";
/// Maximum item-by-quadrature probability cells admitted by LD diagnostics.
pub const LD_MAX_PROBABILITY_CELLS: usize = 20_000_000;
/// Maximum number of upper-triangle LD pair outputs.
pub const LD_MAX_PAIR_OUTPUTS: usize = 5_000_000;
/// Maximum pair-by-person cells traversed by LD diagnostics.
pub const LD_MAX_PAIR_PERSON_CELLS: usize = 200_000_000;
/// Maximum pair-by-quadrature cells traversed by LD diagnostics.
pub const LD_MAX_PAIR_QUADRATURE_CELLS: usize = 200_000_000;

/// Exact work surfaces admitted for one native LD request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LdResourceUsage {
    pub probability_cells: usize,
    pub pair_outputs: usize,
    pub pair_person_cells: usize,
    pub pair_quadrature_cells: usize,
}

fn ld_pair_count(n_items: usize) -> Result<usize, String> {
    if n_items < 2 {
        return Err("local-dependence indices need at least 2 items".into());
    }
    let (left, right) = if n_items % 2 == 0 {
        (n_items / 2, n_items - 1)
    } else {
        (n_items, (n_items - 1) / 2)
    };
    left.checked_mul(right)
        .ok_or_else(|| "local-dependence pair count overflows usize".to_string())
}

fn ld_xi_node_count(bank: &ItemBank<'_>, xi_rule: XiRule) -> Result<usize, String> {
    let (_, uses_space) = crate::model_exec_flags(bank.model_type);
    if !uses_space {
        return Ok(1);
    }
    if !(1..=MAX_XI_LATENT_DIM).contains(&bank.latent_dim) {
        return Err(format!(
            "latent_dim must be in 1..={MAX_XI_LATENT_DIM} for latent-space nodes"
        ));
    }
    match xi_rule {
        XiRule::GaussHermite { q_xi } => {
            crate::quadrature::require_gh_rule(q_xi, "latent-space quadrature size")?;
            if bank.latent_dim > 3 {
                return Err(
                    "tensor Gauss-Hermite supports latent_dim <= 3; use Halton/MonteCarlo".into(),
                );
            }
            q_xi
                .checked_pow(bank.latent_dim as u32)
                .ok_or_else(|| "Gauss-Hermite node count overflows usize".to_string())
        }
        XiRule::Halton { n, .. } => {
            if n == 0 {
                return Err("Halton rule needs n >= 1".into());
            }
            if n > MAX_XI_POINTS {
                return Err(format!(
                    "Halton rule supports at most {MAX_XI_POINTS} points; got {n}"
                ));
            }
            if bank.latent_dim > 6 {
                return Err("Halton rule supports latent_dim <= 6".into());
            }
            Ok(n)
        }
        XiRule::MonteCarlo { n, .. } => {
            if n == 0 {
                return Err("MonteCarlo rule needs n >= 1".into());
            }
            if n > MAX_XI_POINTS {
                return Err(format!(
                    "MonteCarlo rule supports at most {MAX_XI_POINTS} points; got {n}"
                ));
            }
            Ok(n)
        }
    }
}

/// Validate native LD work before `icc_nodes` or pair-vector allocation.
///
/// This is public so Rust consumers can inspect the exact admitted work surface
/// without executing the diagnostic. [`ld_indices`] always calls this same
/// function before delegating to the numerical implementation.
pub fn ld_resource_preflight(
    bank: &ItemBank<'_>,
    n_persons: usize,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<LdResourceUsage, String> {
    let n_items = validate_bank(bank)?;
    let pair_outputs = ld_pair_count(n_items)?;
    if pair_outputs > LD_MAX_PAIR_OUTPUTS {
        return Err(format!(
            "ld_indices pair output exceeds the {LD_MAX_PAIR_OUTPUTS}-pair limit"
        ));
    }

    crate::quadrature::require_gh_rule(q_theta, "quadrature size")?;
    let n_x = ld_xi_node_count(bank, xi_rule)?;
    let quadrature_cells = q_theta
        .checked_mul(n_x)
        .ok_or_else(|| "ld_indices quadrature cell count overflows usize".to_string())?;

    if quadrature_cells > LD_MAX_PROBABILITY_CELLS / n_items {
        return Err(format!(
            "ld_indices workspace exceeds the {LD_MAX_PROBABILITY_CELLS}-probability-cell limit"
        ));
    }
    let probability_cells = n_items * quadrature_cells;

    if n_persons != 0 && pair_outputs > LD_MAX_PAIR_PERSON_CELLS / n_persons {
        return Err(format!(
            "ld_indices pair-person work exceeds the {LD_MAX_PAIR_PERSON_CELLS}-cell limit"
        ));
    }
    let pair_person_cells = pair_outputs * n_persons;

    if quadrature_cells != 0
        && pair_outputs > LD_MAX_PAIR_QUADRATURE_CELLS / quadrature_cells
    {
        return Err(format!(
            "ld_indices pair-quadrature work exceeds the {LD_MAX_PAIR_QUADRATURE_CELLS}-cell limit"
        ));
    }
    let pair_quadrature_cells = pair_outputs * quadrature_cells;

    Ok(LdResourceUsage {
        probability_cells,
        pair_outputs,
        pair_person_cells,
        pair_quadrature_cells,
    })
}

/// Chen-Thissen local-dependence indices with canonical Rust resource admission.
#[allow(clippy::too_many_arguments)]
pub fn ld_indices(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<LdIndexResult, String> {
    ld_resource_preflight(bank, n_persons, q_theta, xi_rule)?;
    implementation::ld_indices(
        bank,
        y,
        observed,
        n_persons,
        prior,
        q_theta,
        xi_rule,
    )
}
