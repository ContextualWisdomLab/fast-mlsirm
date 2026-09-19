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

#[test]
fn one_way_random_intercept_icc_recovers_known_variance_ratio_with_monte_carlo_uncertainty() {
    let mut rng = DeterministicNormal::new(0x5eed_1234_5678_9abc);
    let mut estimates = Vec::with_capacity(REPLICATIONS);

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

        estimates.push(
            one_way_random_intercept_icc(&cluster_ids, &outcomes)
                .expect("identified generated one-way random-intercept design")
                .icc,
        );
    }

    let replication_count = REPLICATIONS as f64;
    let mean = estimates.iter().sum::<f64>() / replication_count;
    let bias = mean - TRUE_ICC;
    let rmse = (estimates
        .iter()
        .map(|estimate| (estimate - TRUE_ICC).powi(2))
        .sum::<f64>()
        / replication_count)
        .sqrt();
    let sampling_variance = estimates
        .iter()
        .map(|estimate| (estimate - mean).powi(2))
        .sum::<f64>()
        / (replication_count - 1.0);
    let monte_carlo_standard_error = (sampling_variance / replication_count).sqrt();

    assert!(
        bias.abs() <= (3.0 * monte_carlo_standard_error).max(0.03),
        "absolute bias {bias} exceeds Monte Carlo acceptance; MCSE={monte_carlo_standard_error}"
    );
    assert!(
        rmse < 0.08,
        "RMSE {rmse} exceeds the true-parameter recovery threshold"
    );
}