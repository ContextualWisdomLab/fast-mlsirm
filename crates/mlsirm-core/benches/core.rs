//! CodSpeed/divan performance benchmarks for the `mlsirm-core` numeric kernels.
//!
//! Inputs are generated deterministically (fixed-seed LCG) so every run measures
//! identical work. Every benchmark pins `Device::Cpu` where a device is
//! selectable, keeping the measurements hardware-independent.

use divan::{black_box, Bencher};
use mlsirm_core::fitstats::infit_outfit;
use mlsirm_core::marginal::{fit_marginal, MarginalConfig, PopulationSpec};
use mlsirm_core::mmle::{fit_mmle_2pl, MmleConfig};
use mlsirm_core::nodes::XiRule;
use mlsirm_core::scoring::{lord_wingersky, score_eap_device, ItemBank, PriorSpec};
use mlsirm_core::{
    neg_loglik_and_grad_device, Device, ModelConfig, ModelType, Params, PenaltyConfig,
};

fn main() {
    divan::main();
}

/// Small deterministic pseudo-random generator (no external dependency).
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1))
    }

    /// Uniform draw in `[0, 1)`.
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Approximately standard-normal draw (Irwin-Hall with 12 uniforms).
    fn normal(&mut self) -> f64 {
        (0..12).map(|_| self.uniform()).sum::<f64>() - 6.0
    }

    fn normals(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| scale * self.normal()).collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A simulated simple-structure MLS2PLM data set with its generating parameters.
struct Fixture {
    config: ModelConfig,
    factor_id: Vec<usize>,
    params: Params,
    y: Vec<f64>,
    observed: Vec<bool>,
}

fn fixture(
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    latent_dim: usize,
    model_type: ModelType,
    missing_rate: f64,
) -> Fixture {
    let mut rng = Lcg::new((n_persons * 7919 + n_items * 104_729 + latent_dim) as u64);
    let factor_id: Vec<usize> = (0..n_items).map(|i| i * n_dims / n_items).collect();
    let params = Params {
        theta: rng.normals(n_persons * n_dims, 1.0),
        alpha: rng.normals(n_items, 0.2),
        b: rng.normals(n_items, 0.8),
        xi: rng.normals(n_persons * latent_dim, 1.0),
        zeta: rng.normals(n_items * latent_dim, 1.0),
        tau: 0.0,
    };
    let gamma = params.tau.exp();
    let mut y = vec![0.0; n_persons * n_items];
    let mut observed = vec![true; n_persons * n_items];
    for p in 0..n_persons {
        for (i, &d) in factor_id.iter().enumerate() {
            let mut dist2 = 0.0;
            for k in 0..latent_dim {
                let diff = params.xi[p * latent_dim + k] - params.zeta[i * latent_dim + k];
                dist2 += diff * diff;
            }
            let eta = params.alpha[i].exp() * params.theta[p * n_dims + d] + params.b[i]
                - gamma * dist2.sqrt();
            let idx = p * n_items + i;
            y[idx] = if rng.uniform() < sigmoid(eta) {
                1.0
            } else {
                0.0
            };
            observed[idx] = rng.uniform() >= missing_rate;
        }
    }
    Fixture {
        config: ModelConfig {
            n_persons,
            n_items,
            n_dims,
            latent_dim,
            model_type,
            eps_distance: 1e-8,
        },
        factor_id,
        params,
        y,
        observed,
    }
}

fn item_bank(f: &Fixture) -> ItemBank<'_> {
    ItemBank {
        alpha: &f.params.alpha,
        b: &f.params.b,
        zeta: &f.params.zeta,
        tau: f.params.tau,
        factor_id: &f.factor_id,
        model_type: f.config.model_type,
        n_dims: f.config.n_dims,
        latent_dim: f.config.latent_dim,
        eps_distance: f.config.eps_distance,
    }
}

/// Joint-likelihood objective and analytic gradients: the JML optimizer hot path.
mod neg_loglik_and_grad {
    use super::*;

    /// `(n_persons, n_items)` problem sizes.
    const SIZES: &[(usize, usize)] = &[(100, 20), (500, 40)];

    #[divan::bench(args = SIZES)]
    fn mls2plm(bencher: Bencher, size: (usize, usize)) {
        let f = fixture(size.0, size.1, 2, 2, ModelType::Mls2plm, 0.0);
        let penalty = PenaltyConfig::default();
        bencher.bench_local(|| {
            neg_loglik_and_grad_device(
                Device::Cpu,
                black_box(&f.y),
                None,
                &f.factor_id,
                black_box(&f.params),
                &f.config,
                &penalty,
            )
        });
    }

    #[divan::bench(args = SIZES)]
    fn mls2plm_masked(bencher: Bencher, size: (usize, usize)) {
        let f = fixture(size.0, size.1, 2, 2, ModelType::Mls2plm, 0.2);
        let penalty = PenaltyConfig::default();
        bencher.bench_local(|| {
            neg_loglik_and_grad_device(
                Device::Cpu,
                black_box(&f.y),
                Some(&f.observed),
                &f.factor_id,
                black_box(&f.params),
                &f.config,
                &penalty,
            )
        });
    }

    #[divan::bench(args = SIZES)]
    fn mirt(bencher: Bencher, size: (usize, usize)) {
        let f = fixture(size.0, size.1, 2, 2, ModelType::Mirt, 0.0);
        let penalty = PenaltyConfig::default();
        bencher.bench_local(|| {
            neg_loglik_and_grad_device(
                Device::Cpu,
                black_box(&f.y),
                None,
                &f.factor_id,
                black_box(&f.params),
                &f.config,
                &penalty,
            )
        });
    }
}

/// Marginal (MMLE-EM) calibration with a bounded number of EM iterations.
mod calibration {
    use super::*;

    #[divan::bench]
    fn mmle_2pl(bencher: Bencher) {
        let f = fixture(300, 20, 1, 1, ModelType::Mirt, 0.1);
        let cfg = MmleConfig {
            max_iter: 10,
            tol: 0.0,
            ..MmleConfig::default()
        };
        bencher
            .bench_local(|| fit_mmle_2pl(black_box(&f.y), black_box(&f.observed), 300, 20, &cfg));
    }

    #[divan::bench]
    fn marginal_mls2plm(bencher: Bencher) {
        let f = fixture(200, 12, 1, 1, ModelType::Mls2plm, 0.0);
        let mcfg = MarginalConfig {
            q_theta: 11,
            q_xi: 7,
            max_iter: 3,
            tol: 0.0,
            m_steps: 2,
            ..MarginalConfig::default()
        };
        let penalty = PenaltyConfig::lsirm_prior();
        bencher.bench_local(|| {
            fit_marginal(
                black_box(&f.y),
                black_box(&f.observed),
                &f.factor_id,
                &f.config,
                &PopulationSpec::Single,
                &mcfg,
                &penalty,
                Device::Cpu,
            )
            .expect("marginal fit must succeed")
        });
    }
}

/// Person scoring and fit statistics against a frozen item bank.
mod scoring {
    use super::*;

    #[divan::bench]
    fn eap_mls2plm(bencher: Bencher) {
        let f = fixture(100, 20, 2, 2, ModelType::Mls2plm, 0.1);
        let bank = item_bank(&f);
        let prior = PriorSpec::standard(2);
        bencher.bench_local(|| {
            score_eap_device(
                &bank,
                black_box(&f.y),
                black_box(&f.observed),
                100,
                &prior,
                15,
                XiRule::GaussHermite { q_xi: 7 },
                Device::Cpu,
            )
            .expect("EAP scoring must succeed")
        });
    }

    #[divan::bench]
    fn lord_wingersky_recursion(bencher: Bencher) {
        let (n_items, n_nodes) = (60, 41);
        let mut rng = Lcg::new(1984);
        let probs: Vec<f64> = (0..n_items * n_nodes)
            .map(|_| 0.05 + 0.9 * rng.uniform())
            .collect();
        bencher.bench_local(|| lord_wingersky(black_box(&probs), n_items, n_nodes));
    }

    #[divan::bench]
    fn infit_outfit_mls2plm(bencher: Bencher) {
        let f = fixture(1_000, 40, 2, 2, ModelType::Mls2plm, 0.1);
        let bank = item_bank(&f);
        bencher.bench_local(|| {
            infit_outfit(
                &bank,
                black_box(&f.y),
                black_box(&f.observed),
                1_000,
                &f.params.theta,
                &f.params.xi,
            )
            .expect("infit/outfit must succeed")
        });
    }
}
