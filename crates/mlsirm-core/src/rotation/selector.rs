//! Criterion-neutral empirical selection across rotation families.
//!
//! Criterion objective values are not commensurable.  This selector therefore
//! ranks candidate solutions on neutral diagnostics: soft row complexity,
//! factor-size collapse, factor-correlation degeneracy, convergence and local
//! basin support, plus signed-permutation-aligned bootstrap congruence when
//! bootstrap loading matrices are supplied.  Selection is policy-conditional,
//! not a claim that one rotation criterion is universally optimal.

use super::{
    rotate_factor_loadings, RotationConfig, RotationCriterion, RotationSolution,
};
use std::cmp::Ordering;

/// Decision policy for aggregating criterion-neutral evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationSelectionPolicy {
    /// Prefer sparse, balanced, readily interpreted simple structure.
    InterpretabilityFirst,
    /// Prefer bootstrap and local-basin reproducibility.
    StabilityFirst,
    /// Prefer agreement with a supplied theory target, then stability.
    TheoryGuided,
    /// Balance simple structure, stability, and degeneracy without a target.
    FullyExploratory,
    /// Prefer recovery against a supplied target; falls back to stability only
    /// when explicitly permitted by a missing target warning.
    RecoveryFirst,
    /// Prefer sparse cross-loading structure with a modest stability guard.
    SparseSimpleStructure,
    /// Prefer bifactor candidates while guarding group-factor collapse.
    BifactorDiscovery,
}

impl RotationSelectionPolicy {
    /// Parse a stable public policy identifier.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "interpretability_first" => Some(Self::InterpretabilityFirst),
            "stability_first" => Some(Self::StabilityFirst),
            "theory_guided" => Some(Self::TheoryGuided),
            "fully_exploratory" => Some(Self::FullyExploratory),
            "recovery_first" => Some(Self::RecoveryFirst),
            "sparse_simple_structure" => Some(Self::SparseSimpleStructure),
            "bifactor_discovery" => Some(Self::BifactorDiscovery),
            _ => None,
        }
    }

    /// Stable public identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InterpretabilityFirst => "interpretability_first",
            Self::StabilityFirst => "stability_first",
            Self::TheoryGuided => "theory_guided",
            Self::FullyExploratory => "fully_exploratory",
            Self::RecoveryFirst => "recovery_first",
            Self::SparseSimpleStructure => "sparse_simple_structure",
            Self::BifactorDiscovery => "bifactor_discovery",
        }
    }
}

/// Neutral metrics and policy score for one candidate criterion.
#[derive(Clone, Debug)]
pub struct RotationCandidateEvidence {
    /// Stable criterion identifier.
    pub criterion_name: &'static str,
    /// Rotated reference-sample solution.
    pub solution: RotationSolution,
    /// Soft row complexity; lower means less cross-loading energy.
    pub row_complexity: f64,
    /// Ratio of smallest to largest column sum of squares; one is balanced.
    pub factor_balance: f64,
    /// Maximum absolute off-diagonal factor correlation.
    pub max_factor_correlation: f64,
    /// Fraction of starts reaching the projected-gradient tolerance.
    pub convergence_rate: f64,
    /// Fraction of starts in the best observed objective basin.
    pub basin_support_rate: f64,
    /// Mean signed-permutation-aligned bootstrap Tucker congruence.
    pub bootstrap_congruence: f64,
    /// Minimum factor-level bootstrap congruence averaged over replicates.
    pub bootstrap_min_congruence: f64,
    /// Root-mean-square deviation from a supplied theory target after alignment.
    pub target_rmse: f64,
    /// Policy-specific normalized rank score; lower is preferred.
    pub policy_score: f64,
    /// Whether no other candidate weakly dominates this one on every active
    /// neutral objective and strictly improves at least one.
    pub pareto_optimal: bool,
}

/// Output of empirical criterion selection.
#[derive(Clone, Debug)]
pub struct RotationSelectionResult {
    /// Selected candidate index in `candidates`.
    pub selected_index: usize,
    /// Selected criterion identifier.
    pub selected_criterion: &'static str,
    /// Decision policy.
    pub policy: RotationSelectionPolicy,
    /// Per-candidate evidence in caller order.
    pub candidates: Vec<RotationCandidateEvidence>,
    /// Number of bootstrap loading matrices used.
    pub bootstrap_replicates: usize,
    /// Evidence grade: `single_sample_diagnostic`, `bootstrap_exploratory`, or
    /// `bootstrap_supported`.
    pub evidence_grade: &'static str,
    /// Explicit interpretation warning.
    pub warning: String,
}

/// Select among already constructed criteria with criterion-neutral evidence.
///
/// `bootstrap_loadings` is a flattened sequence of row-major `rows x factors`
/// loading matrices.  All matrices must be extracted under the same factor
/// count and identification convention as `loadings`.
pub fn select_rotation_criterion(
    loadings: &[f64],
    rows: usize,
    factors: usize,
    candidates: &[RotationCriterion],
    config: &RotationConfig,
    policy: RotationSelectionPolicy,
    bootstrap_loadings: &[Vec<f64>],
    theory_target: Option<&[f64]>,
) -> Result<RotationSelectionResult, String> {
    if candidates.len() < 2 {
        return Err("criterion selection requires at least two candidates".into());
    }
    if let Some(target) = theory_target {
        if target.len() != rows * factors || target.iter().any(|x| !x.is_finite() && !x.is_nan()) {
            return Err("theory_target must match the loading shape and contain finite values or NaN".into());
        }
    }
    if matches!(policy, RotationSelectionPolicy::TheoryGuided) && theory_target.is_none() {
        return Err("theory_guided selection requires theory_target".into());
    }
    if matches!(policy, RotationSelectionPolicy::BifactorDiscovery) && factors < 3 {
        return Err("bifactor_discovery requires at least three factors".into());
    }
    for replicate in bootstrap_loadings {
        if replicate.len() != rows * factors || replicate.iter().any(|x| !x.is_finite()) {
            return Err("every bootstrap loading matrix must be finite and match the reference shape".into());
        }
    }

    let mut evidence = Vec::with_capacity(candidates.len());
    for criterion in candidates {
        let solution = rotate_factor_loadings(loadings, rows, factors, criterion, config)?;
        let (row_complexity, factor_balance) = simple_structure_metrics(
            &solution.pattern_matrix,
            rows,
            factors,
            criterion.fixes_general_factor(),
        );
        let (bootstrap_congruence, bootstrap_min_congruence) = bootstrap_stability(
            &solution,
            criterion,
            config,
            bootstrap_loadings,
            rows,
            factors,
        )?;
        let target_rmse = theory_target
            .map(|target| aligned_target_rmse(&solution.pattern_matrix, target, rows, factors))
            .transpose()?
            .unwrap_or(f64::NAN);
        evidence.push(RotationCandidateEvidence {
            criterion_name: criterion.name(),
            max_factor_correlation: solution.max_factor_correlation,
            convergence_rate: solution.converged_starts as f64 / solution.n_starts as f64,
            basin_support_rate: solution.basin_support as f64 / solution.n_starts as f64,
            solution,
            row_complexity,
            factor_balance,
            bootstrap_congruence,
            bootstrap_min_congruence,
            target_rmse,
            policy_score: f64::NAN,
            pareto_optimal: false,
        });
    }

    assign_pareto_frontier(&mut evidence, theory_target.is_some(), bootstrap_loadings.is_empty());
    assign_policy_scores(
        &mut evidence,
        policy,
        theory_target.is_some(),
        bootstrap_loadings.is_empty(),
    );
    let selected_index = evidence
        .iter()
        .enumerate()
        .min_by(|(index_a, a), (index_b, b)| {
            a.policy_score
                .partial_cmp(&b.policy_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| b.pareto_optimal.cmp(&a.pareto_optimal))
                .then_with(|| index_a.cmp(index_b))
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "criterion selection produced no candidate evidence".to_string())?;
    let bootstrap_replicates = bootstrap_loadings.len();
    let evidence_grade = if bootstrap_replicates == 0 {
        "single_sample_diagnostic"
    } else if bootstrap_replicates < 20 {
        "bootstrap_exploratory"
    } else {
        "bootstrap_supported"
    };
    let warning = if bootstrap_replicates == 0 {
        "Selection is conditional on the requested policy and one loading matrix; provide bootstrap loading matrices before treating the result as stability evidence."
    } else {
        "Selection is conditional on the candidate set, extraction model, bootstrap design, and requested policy; it is not a universal or mathematically proven global optimum."
    }
    .to_string();
    Ok(RotationSelectionResult {
        selected_index,
        selected_criterion: evidence[selected_index].criterion_name,
        policy,
        candidates: evidence,
        bootstrap_replicates,
        evidence_grade,
        warning,
    })
}

fn simple_structure_metrics(
    pattern: &[f64],
    rows: usize,
    factors: usize,
    skip_general: bool,
) -> (f64, f64) {
    let first = usize::from(skip_general);
    let active = factors - first;
    let mut complexity_numerator = 0.0;
    let mut complexity_denominator = 0.0;
    let mut factor_ss = vec![0.0; active];
    for i in 0..rows {
        let row = &pattern[i * factors + first..(i + 1) * factors];
        let row_ss: f64 = row.iter().map(|x| x * x).sum();
        let row_fourth: f64 = row.iter().map(|x| x.powi(4)).sum();
        complexity_numerator += row_ss * row_ss - row_fourth;
        complexity_denominator += row_ss * row_ss;
        for (j, value) in row.iter().enumerate() {
            factor_ss[j] += value * value;
        }
    }
    let row_complexity = if complexity_denominator > 0.0 {
        complexity_numerator / complexity_denominator
    } else {
        1.0
    };
    let minimum = factor_ss.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = factor_ss.iter().copied().fold(0.0_f64, f64::max);
    let factor_balance = if maximum > 0.0 { minimum / maximum } else { 0.0 };
    (row_complexity, factor_balance)
}

fn bootstrap_stability(
    reference: &RotationSolution,
    criterion: &RotationCriterion,
    config: &RotationConfig,
    bootstrap_loadings: &[Vec<f64>],
    rows: usize,
    factors: usize,
) -> Result<(f64, f64), String> {
    if bootstrap_loadings.is_empty() {
        return Ok((f64::NAN, f64::NAN));
    }
    let mut mean_total = 0.0;
    let mut min_total = 0.0;
    for replicate in bootstrap_loadings {
        let rotated = rotate_factor_loadings(replicate, rows, factors, criterion, config)?;
        let congruence = aligned_column_congruence(
            &reference.pattern_matrix,
            &rotated.pattern_matrix,
            rows,
            factors,
            criterion.fixes_general_factor(),
        )?;
        mean_total += congruence.iter().sum::<f64>() / factors as f64;
        min_total += congruence.iter().copied().fold(1.0_f64, f64::min);
    }
    let count = bootstrap_loadings.len() as f64;
    Ok((mean_total / count, min_total / count))
}

fn aligned_target_rmse(
    pattern: &[f64],
    target: &[f64],
    rows: usize,
    factors: usize,
) -> Result<f64, String> {
    let aligned = align_to_reference(pattern, target, rows, factors, false)?;
    let mut sum = 0.0;
    let mut count = 0_usize;
    for (value, target_value) in aligned.iter().zip(target) {
        if target_value.is_nan() {
            continue;
        }
        sum += (value - target_value).powi(2);
        count += 1;
    }
    if count == 0 {
        return Err("theory_target contains no specified cells".into());
    }
    Ok((sum / count as f64).sqrt())
}

fn aligned_column_congruence(
    reference: &[f64],
    candidate: &[f64],
    rows: usize,
    factors: usize,
    fixes_general: bool,
) -> Result<Vec<f64>, String> {
    let aligned = align_to_reference(candidate, reference, rows, factors, fixes_general)?;
    let mut out = vec![0.0; factors];
    for j in 0..factors {
        let mut dot = 0.0;
        let mut ss_reference = 0.0;
        let mut ss_aligned = 0.0;
        for i in 0..rows {
            let a = reference[i * factors + j];
            let b = aligned[i * factors + j];
            dot += a * b;
            ss_reference += a * a;
            ss_aligned += b * b;
        }
        if ss_reference <= 1e-24 || ss_aligned <= 1e-24 {
            return Err("cannot compute congruence for a collapsed factor".into());
        }
        out[j] = (dot / (ss_reference * ss_aligned).sqrt()).abs();
    }
    Ok(out)
}

fn align_to_reference(
    candidate: &[f64],
    reference: &[f64],
    rows: usize,
    factors: usize,
    fixes_general: bool,
) -> Result<Vec<f64>, String> {
    if candidate.len() != rows * factors || reference.len() != rows * factors {
        return Err("alignment matrices must share the declared shape".into());
    }
    let first = usize::from(fixes_general);
    let mut assignment = vec![usize::MAX; factors];
    let mut used = vec![false; factors];
    if fixes_general {
        assignment[0] = 0;
        used[0] = true;
    }
    for reference_column in first..factors {
        let mut best_column = None;
        let mut best_absolute = -1.0_f64;
        for candidate_column in first..factors {
            if used[candidate_column] {
                continue;
            }
            let correlation = raw_column_congruence(
                reference,
                candidate,
                rows,
                factors,
                reference_column,
                candidate_column,
            )?;
            if correlation.abs() > best_absolute {
                best_absolute = correlation.abs();
                best_column = Some(candidate_column);
            }
        }
        let selected = best_column.ok_or_else(|| "factor alignment assignment failed".to_string())?;
        assignment[reference_column] = selected;
        used[selected] = true;
    }
    let mut aligned = vec![0.0; candidate.len()];
    for reference_column in 0..factors {
        let candidate_column = assignment[reference_column];
        let congruence = raw_column_congruence(
            reference,
            candidate,
            rows,
            factors,
            reference_column,
            candidate_column,
        )?;
        let sign = if congruence < 0.0 { -1.0 } else { 1.0 };
        for i in 0..rows {
            aligned[i * factors + reference_column] =
                sign * candidate[i * factors + candidate_column];
        }
    }
    Ok(aligned)
}

fn raw_column_congruence(
    reference: &[f64],
    candidate: &[f64],
    rows: usize,
    factors: usize,
    reference_column: usize,
    candidate_column: usize,
) -> Result<f64, String> {
    let mut dot = 0.0;
    let mut ss_reference = 0.0;
    let mut ss_candidate = 0.0;
    for i in 0..rows {
        let a = reference[i * factors + reference_column];
        let b = candidate[i * factors + candidate_column];
        dot += a * b;
        ss_reference += a * a;
        ss_candidate += b * b;
    }
    if ss_reference <= 1e-24 || ss_candidate <= 1e-24 {
        return Err("factor alignment encountered a collapsed column".into());
    }
    Ok(dot / (ss_reference * ss_candidate).sqrt())
}

fn assign_pareto_frontier(
    evidence: &mut [RotationCandidateEvidence],
    has_target: bool,
    no_bootstrap: bool,
) {
    for index in 0..evidence.len() {
        evidence[index].pareto_optimal = !(0..evidence.len()).any(|other| {
            other != index
                && dominates(&evidence[other], &evidence[index], has_target, no_bootstrap)
        });
    }
}

fn dominates(
    a: &RotationCandidateEvidence,
    b: &RotationCandidateEvidence,
    has_target: bool,
    no_bootstrap: bool,
) -> bool {
    let mut all_no_worse = a.row_complexity <= b.row_complexity
        && a.factor_balance >= b.factor_balance
        && a.max_factor_correlation <= b.max_factor_correlation
        && a.convergence_rate >= b.convergence_rate
        && a.basin_support_rate >= b.basin_support_rate;
    let mut strictly_better = a.row_complexity < b.row_complexity
        || a.factor_balance > b.factor_balance
        || a.max_factor_correlation < b.max_factor_correlation
        || a.convergence_rate > b.convergence_rate
        || a.basin_support_rate > b.basin_support_rate;
    if !no_bootstrap {
        all_no_worse &= a.bootstrap_congruence >= b.bootstrap_congruence
            && a.bootstrap_min_congruence >= b.bootstrap_min_congruence;
        strictly_better |= a.bootstrap_congruence > b.bootstrap_congruence
            || a.bootstrap_min_congruence > b.bootstrap_min_congruence;
    }
    if has_target {
        all_no_worse &= a.target_rmse <= b.target_rmse;
        strictly_better |= a.target_rmse < b.target_rmse;
    }
    all_no_worse && strictly_better
}

fn assign_policy_scores(
    evidence: &mut [RotationCandidateEvidence],
    policy: RotationSelectionPolicy,
    has_target: bool,
    no_bootstrap: bool,
) {
    let complexity = ranks(evidence.iter().map(|x| x.row_complexity).collect(), false);
    let balance = ranks(evidence.iter().map(|x| x.factor_balance).collect(), true);
    let correlation = ranks(
        evidence.iter().map(|x| x.max_factor_correlation).collect(),
        false,
    );
    let convergence = ranks(evidence.iter().map(|x| x.convergence_rate).collect(), true);
    let basin = ranks(evidence.iter().map(|x| x.basin_support_rate).collect(), true);
    let bootstrap = if no_bootstrap {
        vec![0.0; evidence.len()]
    } else {
        ranks(
            evidence.iter().map(|x| x.bootstrap_congruence).collect(),
            true,
        )
    };
    let bootstrap_min = if no_bootstrap {
        vec![0.0; evidence.len()]
    } else {
        ranks(
            evidence
                .iter()
                .map(|x| x.bootstrap_min_congruence)
                .collect(),
            true,
        )
    };
    let target = if has_target {
        ranks(evidence.iter().map(|x| x.target_rmse).collect(), false)
    } else {
        vec![0.0; evidence.len()]
    };
    for index in 0..evidence.len() {
        let stability = if no_bootstrap {
            0.55 * basin[index] + 0.45 * convergence[index]
        } else {
            0.45 * bootstrap[index]
                + 0.30 * bootstrap_min[index]
                + 0.15 * basin[index]
                + 0.10 * convergence[index]
        };
        evidence[index].policy_score = match policy {
            RotationSelectionPolicy::InterpretabilityFirst => {
                0.50 * complexity[index]
                    + 0.18 * balance[index]
                    + 0.17 * correlation[index]
                    + 0.15 * stability
            }
            RotationSelectionPolicy::StabilityFirst => {
                0.65 * stability + 0.20 * correlation[index] + 0.15 * complexity[index]
            }
            RotationSelectionPolicy::TheoryGuided => {
                0.60 * target[index] + 0.25 * stability + 0.15 * complexity[index]
            }
            RotationSelectionPolicy::FullyExploratory => {
                0.35 * complexity[index]
                    + 0.30 * stability
                    + 0.20 * correlation[index]
                    + 0.15 * balance[index]
            }
            RotationSelectionPolicy::RecoveryFirst => {
                if has_target {
                    0.65 * target[index] + 0.25 * stability + 0.10 * correlation[index]
                } else {
                    0.65 * stability + 0.25 * complexity[index] + 0.10 * correlation[index]
                }
            }
            RotationSelectionPolicy::SparseSimpleStructure => {
                0.65 * complexity[index]
                    + 0.15 * balance[index]
                    + 0.10 * stability
                    + 0.10 * correlation[index]
            }
            RotationSelectionPolicy::BifactorDiscovery => {
                let bifactor_penalty = if matches!(
                    evidence[index].criterion_name,
                    "bifactor" | "bigeomin"
                ) {
                    0.0
                } else {
                    1.0
                };
                0.45 * complexity[index]
                    + 0.20 * balance[index]
                    + 0.15 * stability
                    + 0.10 * correlation[index]
                    + 0.10 * bifactor_penalty
            }
        };
    }
}

fn ranks(values: Vec<f64>, descending: bool) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| {
        let comparison = values[*a]
            .partial_cmp(&values[*b])
            .unwrap_or(Ordering::Equal);
        if descending {
            comparison.reverse()
        } else {
            comparison
        }
        .then_with(|| a.cmp(b))
    });
    let denominator = values.len().saturating_sub(1).max(1) as f64;
    let mut result = vec![0.0; values.len()];
    for (rank, index) in order.into_iter().enumerate() {
        result[index] = rank as f64 / denominator;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::{RotationMode, RotationSelectionPolicy::*};

    fn reference() -> Vec<f64> {
        vec![0.72, 0.39, 0.65, 0.35, 0.60, 0.31, -0.31, 0.70, -0.28, 0.64, -0.25, 0.58]
    }

    fn config() -> RotationConfig {
        RotationConfig {
            mode: RotationMode::Orthogonal,
            n_starts: 4,
            max_iter: 400,
            tolerance: 1e-6,
            max_threads: 1,
            ..RotationConfig::default()
        }
    }

    #[test]
    fn selector_returns_policy_conditional_pareto_evidence() {
        let candidates = vec![
            RotationCriterion::Varimax,
            RotationCriterion::Quartimax,
            RotationCriterion::CrawfordFerguson { kappa: 0.25 },
        ];
        let bootstraps = vec![
            reference()
                .iter()
                .enumerate()
                .map(|(index, value)| value + 0.005 * (index as f64 - 5.0))
                .collect(),
            reference()
                .iter()
                .enumerate()
                .map(|(index, value)| value - 0.004 * (index as f64 - 5.0))
                .collect(),
        ];
        let result = select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            FullyExploratory,
            &bootstraps,
            None,
        )
        .unwrap();
        assert_eq!(result.candidates.len(), 3);
        assert!(result.candidates.iter().any(|x| x.pareto_optimal));
        assert_eq!(result.evidence_grade, "bootstrap_exploratory");
        assert_eq!(result.bootstrap_replicates, 2);
        assert_eq!(result.policy.as_str(), "fully_exploratory");
        assert_eq!(
            result.selected_criterion,
            result.candidates[result.selected_index].criterion_name
        );
        assert!(result.candidates.iter().all(|x| x.policy_score.is_finite()));
    }

    #[test]
    fn theory_policy_uses_target_and_supported_bootstrap_grade() {
        let candidates = vec![RotationCriterion::Varimax, RotationCriterion::Quartimax];
        let target = vec![0.8, 0.0, 0.7, 0.0, 0.6, 0.0, 0.0, 0.8, 0.0, 0.7, 0.0, 0.6];
        let bootstraps = vec![reference(); 20];
        let result = select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            TheoryGuided,
            &bootstraps,
            Some(&target),
        )
        .unwrap();
        assert_eq!(result.evidence_grade, "bootstrap_supported");
        assert!(result.candidates.iter().all(|x| x.target_rmse.is_finite()));
    }

    #[test]
    fn single_sample_policies_and_validation_fail_closed() {
        let candidates = vec![RotationCriterion::Varimax, RotationCriterion::Quartimax];
        for policy in [
            InterpretabilityFirst,
            StabilityFirst,
            RecoveryFirst,
            SparseSimpleStructure,
        ] {
            let result = select_rotation_criterion(
                &reference(),
                6,
                2,
                &candidates,
                &config(),
                policy,
                &[],
                None,
            )
            .unwrap();
            assert_eq!(result.evidence_grade, "single_sample_diagnostic");
            assert!(result.warning.contains("one loading matrix"));
        }
        assert!(select_rotation_criterion(
            &reference(),
            6,
            2,
            &[RotationCriterion::Varimax],
            &config(),
            FullyExploratory,
            &[],
            None,
        )
        .is_err());
        assert!(select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            TheoryGuided,
            &[],
            None,
        )
        .is_err());
        assert!(select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            BifactorDiscovery,
            &[],
            None,
        )
        .is_err());
        assert!(select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            FullyExploratory,
            &[vec![0.0; 2]],
            None,
        )
        .is_err());
        assert!(select_rotation_criterion(
            &reference(),
            6,
            2,
            &candidates,
            &config(),
            FullyExploratory,
            &[],
            Some(&[0.0; 2]),
        )
        .is_err());
    }

    #[test]
    fn policy_parser_and_alignment_edge_cases_are_covered() {
        for (name, expected) in [
            ("interpretability-first", InterpretabilityFirst),
            ("stability_first", StabilityFirst),
            ("theory_guided", TheoryGuided),
            ("fully_exploratory", FullyExploratory),
            ("recovery_first", RecoveryFirst),
            ("sparse_simple_structure", SparseSimpleStructure),
            ("bifactor_discovery", BifactorDiscovery),
        ] {
            assert_eq!(RotationSelectionPolicy::parse(name), Some(expected));
        }
        assert_eq!(RotationSelectionPolicy::parse("bad"), None);
        assert!(aligned_target_rmse(&[0.5, 0.1, 0.2, 0.4], &[f64::NAN; 4], 2, 2).is_err());
        assert!(aligned_column_congruence(&[0.0; 4], &[0.0; 4], 2, 2, false).is_err());
        assert!(align_to_reference(&[0.0; 2], &[0.0; 4], 2, 2, false).is_err());
    }

    #[test]
    fn bifactor_policy_prefers_bifactor_family_when_other_evidence_ties() {
        let mut evidence = vec![
            mock_evidence("varimax"),
            mock_evidence("bifactor"),
            mock_evidence("bigeomin"),
        ];
        assign_policy_scores(&mut evidence, BifactorDiscovery, false, true);
        assert!(evidence[1].policy_score < evidence[0].policy_score);
        assert!(evidence[2].policy_score < evidence[0].policy_score);
    }

    fn mock_evidence(name: &'static str) -> RotationCandidateEvidence {
        RotationCandidateEvidence {
            criterion_name: name,
            solution: RotationSolution {
                pattern_matrix: vec![0.5, 0.1, 0.2, 0.4],
                structure_matrix: vec![0.5, 0.1, 0.2, 0.4],
                factor_correlation: vec![1.0, 0.0, 0.0, 1.0],
                transform_matrix: vec![1.0, 0.0, 0.0, 1.0],
                n_rows: 2,
                n_factors: 2,
                criterion_name: name,
                mode: RotationMode::Orthogonal,
                criterion_value: 0.0,
                gradient_norm: 0.0,
                iterations: 0,
                converged: true,
                termination_reason: "test",
                best_start_index: 0,
                n_starts: 1,
                converged_starts: 1,
                basin_support: 1,
                distinct_minima: 1,
                start_values: vec![0.0],
                start_converged: vec![true],
                max_factor_correlation: 0.0,
                normalized: false,
                worker_count: 1,
                backend: "test",
            },
            row_complexity: 0.1,
            factor_balance: 0.9,
            max_factor_correlation: 0.0,
            convergence_rate: 1.0,
            basin_support_rate: 1.0,
            bootstrap_congruence: f64::NAN,
            bootstrap_min_congruence: f64::NAN,
            target_rmse: f64::NAN,
            policy_score: f64::NAN,
            pareto_optimal: false,
        }
    }
}