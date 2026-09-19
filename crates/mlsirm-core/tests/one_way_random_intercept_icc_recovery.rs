use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

const REPLICATIONS: usize = 64;
const CLUSTERS_PER_REPLICATION: usize = 120;
const TRUE_BETWEEN_VARIANCE: f64 = 2.0;
const TRUE_WITHIN_VARIANCE: f64 = 1.0;
const TRUE_ICC: f64 = TRUE_BETWEEN_VARIANCE / (TRUE_BETWEEN_VARIANCE + TRUE_WITHIN_VARIANCE);

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

fn ratio_only_recovery_accepts(icc_estimates: &[f64]) -> bool {
    bias_within_monte_carlo_uncertainty(icc_estimates, TRUE_ICC).2
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
fn ratio_only_recovery_rejects_proportionally_wrong_variance_components() {
    let icc_estimates = [TRUE_ICC; 8];
    let between_estimates = [TRUE_BETWEEN_VARIANCE * 2.0; 8];
    let within_estimates = [TRUE_WITHIN_VARIANCE * 2.0; 8];

    let ratio_only_accepted = ratio_only_recovery_accepts(&icc_estimates);
    let between_accepted =
        bias_within_monte_carlo_uncertainty(&between_estimates, TRUE_BETWEEN_VARIANCE).2;
    let within_accepted =
        bias_within_monte_carlo_uncertainty(&within_estimates, TRUE_WITHIN_VARIANCE).2;

    assert!(ratio_only_accepted, "witness must preserve the ICC ratio exactly");
    assert!(!between_accepted && !within_accepted, "witness must miss both component truths");
    assert!(
        !ratio_only_accepted,
        "scientific recovery must not accept the ICC ratio while both variance components are wrong"
    );
}

#[test]
fn one_way_random_intercept_icc_recovers_known_variance_ratio_with_monte_carlo_uncertainty() {
    let mut rng = DeterministicNormal::new(0x5eed_1234_5678_9abc);
    let attempted = REPLICATIONS;
    let mut icc_estimates = Vec::with_capacity(REPLICATIONS);
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
            Ok(result) => icc_estimates.push(result.icc),
            Err(error) => {
                failed += 1;
                if first_failure.is_none() {
                    first_failure = Some(error);
                }
            }
        }
    }

    let recovered = icc_estimates.len();
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

    let replication_count = recovered as f64;
    let mean = icc_estimates.iter().sum::<f64>() / replication_count;
    let bias = mean - TRUE_ICC;
    let rmse = (icc_estimates
        .iter()
        .map(|estimate| (estimate - TRUE_ICC).powi(2))
        .sum::<f64>()
        / replication_count)
        .sqrt();
    let (_, monte_carlo_standard_error, accepted) =
        bias_within_monte_carlo_uncertainty(&icc_estimates, TRUE_ICC);

    assert!(
        ratio_only_recovery_accepts(&icc_estimates) && accepted,
        "absolute bias {bias} exceeds 3*MCSE={}; attempted={attempted}, recovered={recovered}, failed={failed}",
        3.0 * monte_carlo_standard_error
    );
    assert!(
        rmse < 0.08,
        "RMSE {rmse} exceeds the true-parameter recovery threshold; attempted={attempted}, recovered={recovered}, failed={failed}"
    );
}
