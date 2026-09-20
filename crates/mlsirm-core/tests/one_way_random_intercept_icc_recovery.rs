use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

const REPLICATIONS: usize = 256;
const CLUSTERS_PER_REPLICATION: usize = 120;
const TRUE_BETWEEN_VARIANCE: f64 = 2.0;
const TRUE_WITHIN_VARIANCE: f64 = 1.0;
const TRUE_ICC: f64 = TRUE_BETWEEN_VARIANCE / (TRUE_BETWEEN_VARIANCE + TRUE_WITHIN_VARIANCE);
const MAX_ICC_RMSE: f64 = 0.08;
const MAX_COMPONENT_RELATIVE_RMSE: f64 = 0.20;

struct DeterministicNormal {
    state: u64,
}

impl DeterministicNormal {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn uniform_open_zero(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let unit = (self.state >> 11) as f64 * (1.0 / ((1_u64 << 53) as f64));
        unit.max(f64::EPSILON)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform_open_zero();
        let u2 = self.uniform_open_zero();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn bias_within_monte_carlo_uncertainty(estimates: &[f64], truth: f64) -> (f64, f64, bool) {
    assert!(
        estimates.len() >= 2,
        "Monte Carlo bias acceptance requires at least two recovered replications"
    );
    let replication_count = estimates.len() as f64;
    let mean = estimates.iter().sum::<f64>() / replication_count;
    let bias = mean - truth;
    let sampling_variance = estimates
        .iter()
        .map(|estimate| (estimate - mean).powi(2))
        .sum::<f64>()
        / (replication_count - 1.0);
    let monte_carlo_standard_error = (sampling_variance / replication_count).sqrt();
    let accepted = bias.abs() <= 3.0 * monte_carlo_standard_error;
    (bias, monte_carlo_standard_error, accepted)
}

fn root_mean_squared_error(estimates: &[f64], truth: f64) -> f64 {
    (estimates
        .iter()
        .map(|estimate| (estimate - truth).powi(2))
        .sum::<f64>()
        / estimates.len() as f64)
        .sqrt()
}

fn root_mean_squared_error_monte_carlo_standard_error(estimates: &[f64], truth: f64) -> f64 {
    assert!(
        estimates.len() >= 2,
        "RMSE Monte Carlo uncertainty requires at least two recovered replications"
    );
    let replication_count = estimates.len() as f64;
    let squared_errors: Vec<f64> = estimates
        .iter()
        .map(|estimate| (estimate - truth).powi(2))
        .collect();
    let mean_squared_error = squared_errors.iter().sum::<f64>() / replication_count;
    let rmse = mean_squared_error.sqrt();
    if rmse == 0.0 {
        return 0.0;
    }
    let squared_error_variance = squared_errors
        .iter()
        .map(|squared_error| (squared_error - mean_squared_error).powi(2))
        .sum::<f64>()
        / (replication_count - 1.0);
    let mse_monte_carlo_standard_error = (squared_error_variance / replication_count).sqrt();
    mse_monte_carlo_standard_error / (2.0 * rmse)
}

fn rmse_upper_bound(estimates: &[f64], truth: f64) -> (f64, f64, f64) {
    let rmse = root_mean_squared_error(estimates, truth);
    let monte_carlo_standard_error =
        root_mean_squared_error_monte_carlo_standard_error(estimates, truth);
    (
        rmse,
        monte_carlo_standard_error,
        rmse + 3.0 * monte_carlo_standard_error,
    )
}

fn recovery_accepts_all_estimands(
    icc_estimates: &[f64],
    between_estimates: &[f64],
    within_estimates: &[f64],
) -> bool {
    let icc_bias_accepted = bias_within_monte_carlo_uncertainty(icc_estimates, TRUE_ICC).2;
    let between_bias_accepted =
        bias_within_monte_carlo_uncertainty(between_estimates, TRUE_BETWEEN_VARIANCE).2;
    let within_bias_accepted =
        bias_within_monte_carlo_uncertainty(within_estimates, TRUE_WITHIN_VARIANCE).2;

    let (_, _, icc_rmse_upper_bound) = rmse_upper_bound(icc_estimates, TRUE_ICC);
    let (_, _, between_rmse_upper_bound) =
        rmse_upper_bound(between_estimates, TRUE_BETWEEN_VARIANCE);
    let (_, _, within_rmse_upper_bound) =
        rmse_upper_bound(within_estimates, TRUE_WITHIN_VARIANCE);

    icc_bias_accepted
        && between_bias_accepted
        && within_bias_accepted
        && icc_rmse_upper_bound < MAX_ICC_RMSE
        && between_rmse_upper_bound / TRUE_BETWEEN_VARIANCE < MAX_COMPONENT_RELATIVE_RMSE
        && within_rmse_upper_bound / TRUE_WITHIN_VARIANCE < MAX_COMPONENT_RELATIVE_RMSE
}

#[test]
fn bias_acceptance_does_not_replace_monte_carlo_uncertainty_with_an_absolute_floor() {
    let truth = 2.0 / 3.0;
    let estimates = [
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
    ];
    let (bias, monte_carlo_standard_error, accepted) =
        bias_within_monte_carlo_uncertainty(&estimates, truth);

    assert!(bias.abs() < 0.03, "witness must remain below the legacy floor");
    assert!(
        bias.abs() > 3.0 * monte_carlo_standard_error,
        "witness must be outside the declared 3*MCSE acceptance region"
    );
    assert!(
        !accepted,
        "bias {bias} is outside 3*MCSE={}; an unrelated absolute floor must not accept it",
        3.0 * monte_carlo_standard_error
    );
}

#[test]
fn practical_bias_inside_declared_budget_must_not_fail_only_because_mcse_is_small() {
    let truth = TRUE_ICC;
    let estimates = [
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
        truth + 0.019,
        truth + 0.021,
    ];
    let (bias, monte_carlo_standard_error, accepted) =
        bias_within_monte_carlo_uncertainty(&estimates, truth);
    let practical_bias_budget = MAX_ICC_RMSE / 2.0;

    assert!(
        bias.abs() + 3.0 * monte_carlo_standard_error < practical_bias_budget,
        "witness must remain inside the predeclared practical bias budget after Monte Carlo uncertainty"
    );
    assert!(
        accepted,
        "a precisely estimated small bias inside the practical budget must not be rejected merely for being many MCSE from zero"
    );
}

#[test]
fn ratio_only_recovery_rejects_proportionally_wrong_variance_components() {
    let icc_estimates = [TRUE_ICC; 8];
    let between_estimates = [TRUE_BETWEEN_VARIANCE * 2.0; 8];
    let within_estimates = [TRUE_WITHIN_VARIANCE * 2.0; 8];

    let ratio_only_accepted = bias_within_monte_carlo_uncertainty(&icc_estimates, TRUE_ICC).2;
    let all_estimands_accepted =
        recovery_accepts_all_estimands(&icc_estimates, &between_estimates, &within_estimates);

    assert!(ratio_only_accepted, "witness must preserve the ICC ratio exactly");
    assert!(
        !all_estimands_accepted,
        "scientific recovery must reject a correct ratio when both component scales are wrong"
    );
}

#[test]
fn rmse_acceptance_accounts_for_monte_carlo_uncertainty_near_threshold() {
    let icc_estimates = [
        TRUE_ICC,
        TRUE_ICC + 0.1,
        TRUE_ICC,
        TRUE_ICC + 0.1,
        TRUE_ICC,
        TRUE_ICC + 0.1,
        TRUE_ICC,
        TRUE_ICC + 0.1,
    ];
    let between_estimates = [TRUE_BETWEEN_VARIANCE; 8];
    let within_estimates = [TRUE_WITHIN_VARIANCE; 8];
    let (point_rmse, rmse_mcse, rmse_upper_bound) = rmse_upper_bound(&icc_estimates, TRUE_ICC);

    assert!(
        point_rmse < MAX_ICC_RMSE,
        "witness must pass the legacy point-only RMSE gate"
    );
    assert!(
        rmse_upper_bound > MAX_ICC_RMSE,
        "witness uncertainty must cross the declared RMSE target: point={point_rmse}, 3*MCSE={}, upper={rmse_upper_bound}",
        3.0 * rmse_mcse
    );
    assert!(
        !recovery_accepts_all_estimands(&icc_estimates, &between_estimates, &within_estimates),
        "scientific acceptance must reject a point RMSE whose 3*MCSE upper bound crosses the target"
    );
}

#[test]
fn one_way_random_intercept_icc_recovers_known_variance_components_and_ratio() {
    let mut rng = DeterministicNormal::new(0x5eed_1234_5678_9abc);
    let attempted = REPLICATIONS;
    let mut icc_estimates = Vec::with_capacity(REPLICATIONS);
    let mut between_estimates = Vec::with_capacity(REPLICATIONS);
    let mut within_estimates = Vec::with_capacity(REPLICATIONS);
    let mut failed = 0_usize;
    let mut first_failure = None;

    for _ in 0..REPLICATIONS {
        let mut cluster_ids = Vec::new();
        let mut outcomes = Vec::new();
        for cluster_id in 0..CLUSTERS_PER_REPLICATION {
            let random_intercept = TRUE_BETWEEN_VARIANCE.sqrt() * rng.standard_normal();
            let cluster_size = 2 + cluster_id % 4;
            for _ in 0..cluster_size {
                cluster_ids.push(cluster_id as u64);
                outcomes.push(
                    random_intercept + TRUE_WITHIN_VARIANCE.sqrt() * rng.standard_normal(),
                );
            }
        }

        match one_way_random_intercept_icc(&cluster_ids, &outcomes) {
            Ok(result) => {
                icc_estimates.push(result.icc);
                between_estimates.push(result.between_variance);
                within_estimates.push(result.within_variance);
            }
            Err(error) => {
                failed += 1;
                if first_failure.is_none() {
                    first_failure = Some(error);
                }
            }
        }
    }

    let recovered = icc_estimates.len();
    assert_eq!(between_estimates.len(), recovered);
    assert_eq!(within_estimates.len(), recovered);
    assert_eq!(
        attempted,
        recovered + failed,
        "every attempted replication must be accounted for"
    );
    assert_eq!(
        failed, 0,
        "true-parameter recovery failed: attempted={attempted}, recovered={recovered}, failed={failed}, first_failure={first_failure:?}"
    );
    assert_eq!(
        recovered, attempted,
        "scientific acceptance requires a complete recovery denominator"
    );

    let (icc_bias, icc_mcse, icc_bias_accepted) =
        bias_within_monte_carlo_uncertainty(&icc_estimates, TRUE_ICC);
    let (between_bias, between_mcse, between_bias_accepted) =
        bias_within_monte_carlo_uncertainty(&between_estimates, TRUE_BETWEEN_VARIANCE);
    let (within_bias, within_mcse, within_bias_accepted) =
        bias_within_monte_carlo_uncertainty(&within_estimates, TRUE_WITHIN_VARIANCE);

    let (icc_rmse, icc_rmse_mcse, icc_rmse_upper_bound) =
        rmse_upper_bound(&icc_estimates, TRUE_ICC);
    let (between_rmse, between_rmse_mcse, between_rmse_upper_bound) =
        rmse_upper_bound(&between_estimates, TRUE_BETWEEN_VARIANCE);
    let (within_rmse, within_rmse_mcse, within_rmse_upper_bound) =
        rmse_upper_bound(&within_estimates, TRUE_WITHIN_VARIANCE);
    let between_relative_rmse_upper_bound = between_rmse_upper_bound / TRUE_BETWEEN_VARIANCE;
    let within_relative_rmse_upper_bound = within_rmse_upper_bound / TRUE_WITHIN_VARIANCE;

    assert!(
        icc_bias_accepted,
        "ICC bias {icc_bias} exceeds 3*MCSE={}; attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * icc_mcse
    );
    assert!(
        between_bias_accepted,
        "between-variance bias {between_bias} exceeds 3*MCSE={}; attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * between_mcse
    );
    assert!(
        within_bias_accepted,
        "within-variance bias {within_bias} exceeds 3*MCSE={}; attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * within_mcse
    );
    assert!(
        icc_rmse_upper_bound < MAX_ICC_RMSE,
        "ICC RMSE upper bound {icc_rmse_upper_bound} exceeds {MAX_ICC_RMSE}; point={icc_rmse}, 3*MCSE={}, attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * icc_rmse_mcse
    );
    assert!(
        between_relative_rmse_upper_bound < MAX_COMPONENT_RELATIVE_RMSE,
        "between-variance relative RMSE upper bound {between_relative_rmse_upper_bound} exceeds {MAX_COMPONENT_RELATIVE_RMSE}; point={between_rmse}, 3*MCSE={}, attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * between_rmse_mcse
    );
    assert!(
        within_relative_rmse_upper_bound < MAX_COMPONENT_RELATIVE_RMSE,
        "within-variance relative RMSE upper bound {within_relative_rmse_upper_bound} exceeds {MAX_COMPONENT_RELATIVE_RMSE}; point={within_rmse}, 3*MCSE={}, attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * within_rmse_mcse
    );
    assert!(
        recovery_accepts_all_estimands(&icc_estimates, &between_estimates, &within_estimates),
        "joint recovery acceptance must require every returned estimand with Monte Carlo uncertainty"
    );
}
