//! Relative DIF model selection using the existing Rust information-criteria definition.
//!
//! Kang, T., Cohen, A. S., & Sung, H.-J. (2009). Model selection indices for
//! polytomous items. *Applied Psychological Measurement, 33*(7), 499–518.
//! https://doi.org/10.1177/0146621608327800 (pp. 500, 503: relative comparisons,
//! not significance tests; p. 502, eq. 4: posterior DIC, not implemented here).
//! See `docs/dif_information_criteria.md` for the pinned mirt source audit and
//! the distinction from Holm p-value adjustment. Scalar arithmetic is CPU f64.

/// Caller-attested comparable observed-ML scalar evidence; N counts persons.
#[derive(Clone, Debug)]
pub struct DifFitEvidence {
    pub loglik: f64,
    pub n_parameters: usize,
    pub n_persons: usize,
    pub converged: bool,
    pub likelihood_kind: String,
    pub comparison_key: String,
}

/// Relative IC preference, not a hypothesis-test or multiplicity decision.
#[derive(Clone, Debug)]
pub struct DifInformationCriteriaComparison {
    pub criterion: String,
    pub compact_ic: f64,
    pub augmented_ic: f64,
    pub delta_ic: f64,
    pub prefer_augmented: bool,
}

/// Compare nested compact/augmented models by an explicit IC using the existing
/// definitions (Kang et al., 2009, pp. 500, 503; full reference above). No p-value
/// or family-wise error control is provided. Posterior DIC is unsupported.
pub fn compare_dif_information_criteria(
    compact: &DifFitEvidence,
    augmented: &DifFitEvidence,
    criterion: &str,
) -> Result<DifInformationCriteriaComparison, String> {
    for fit in [compact, augmented] {
        if !fit.converged {
            return Err("DIF IC selection requires converged fits".into());
        }
        if !fit.loglik.is_finite() {
            return Err("observed marginal loglik must be finite".into());
        }
        if fit.n_persons < 2 {
            return Err("n_persons must be >= 2".into());
        }
        if fit.likelihood_kind != "unpenalized_marginal_ml" {
            return Err("ordinary ML IC requires unpenalized maximized marginal likelihood, not MAP/posterior objectives".into());
        }
        if fit.comparison_key.trim().is_empty() {
            return Err(
                "comparison_key must bind data, row order, estimator and integration".into(),
            );
        }
    }
    if compact.n_persons != augmented.n_persons
        || compact.comparison_key != augmented.comparison_key
    {
        return Err("fits must share n_persons and comparison_key (data, row order, estimator, integration)".into());
    }
    if augmented.n_parameters <= compact.n_parameters {
        return Err("augmented fit must have more free parameters than compact".into());
    }
    let select = |ic: crate::fitstats::InformationCriteria| -> Result<f64, String> {
        match criterion {
            "AIC" => Ok(ic.aic),
            "BIC" => Ok(ic.bic),
            "AICc" => Ok(ic.aicc),
            "SABIC" => Ok(ic.sabic),
            "CAIC" => Ok(ic.caic),
            "DIC" => Err("DIC is unsupported: posterior deviance and a likelihood-focus contract are required".into()),
            _ => Err("criterion must be AIC, BIC, AICc, SABIC or CAIC; delta_ic is a difference, not DIC".into()),
        }
    };
    let compact_ic = select(crate::fitstats::information_criteria(
        compact.loglik,
        compact.n_parameters,
        compact.n_persons,
    ))?;
    let augmented_ic = select(crate::fitstats::information_criteria(
        augmented.loglik,
        augmented.n_parameters,
        augmented.n_persons,
    ))?;
    let delta_ic = augmented_ic - compact_ic;
    if !compact_ic.is_finite() || !augmented_ic.is_finite() || !delta_ic.is_finite() {
        return Err(
            "criterion must be finite; AICc is undefined when n_persons <= n_parameters + 1".into(),
        );
    }
    Ok(DifInformationCriteriaComparison {
        criterion: criterion.into(),
        compact_ic,
        augmented_ic,
        delta_ic,
        prefer_augmented: delta_ic < 0.0,
    })
}
