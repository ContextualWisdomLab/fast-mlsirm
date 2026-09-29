//! No-prior regression against main 99c228a8 on fixed synthetic inputs.
//!
//! The reference values below were produced by main 99c228a8 (before the
//! slope prior existed) with Rust 1.97.1 on aarch64-apple-darwin, where this
//! branch reproduces them bit for bit. They are compared with a tolerance
//! rather than as a bit hash because `f64::exp`/`ln` come from the platform
//! libm, whose last-ulp rounding differs across libc versions and CPU
//! feature dispatch; an earlier bit hash pinned on one x86_64 host never
//! matched the CI runner. Discrete outcomes (counts, iteration count,
//! termination) are still compared exactly.

use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm, fit_bifactor_grm_multigroup, BifactorGrmConfig, BifactorMultigroupConfig,
    SlopePrior,
};

const TOL: f64 = 1e-8;

fn close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (idx, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= TOL * w.abs().max(1.0),
            "{label}[{idx}] diverged from main 99c228a8: {g} vs {w}"
        );
    }
}

fn data() -> (Vec<usize>, Vec<usize>) {
    let mut state = 20_260_921u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((state >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
    };
    let mut y = Vec::with_capacity(2 * 200 * 6);
    let mut groups = Vec::with_capacity(400);
    let ag = [1.3, 1.1, 0.9, 1.2, 1.0, 0.8];
    let as_ = [1.0, 0.9, 1.1, 0.8, 1.0, 0.9];
    let d = [
        1.2, -0.8, 1.0, -1.0, 1.3, -0.7, 1.1, -0.9, 0.9, -1.1, 1.2, -0.8,
    ];
    for group in 0..2 {
        for _ in 0..200 {
            let (tg, ts) = {
                let mut normal = || {
                    let u1 = uniform().clamp(1e-12, 1.0 - 1e-12);
                    let u2 = uniform();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                (normal() + 0.5 * group as f64, [normal(), normal()])
            };
            for i in 0..6 {
                let base = ag[i] * tg + as_[i] * ts[i / 3];
                let u = uniform();
                let mut cat = 0;
                for k in 0..2 {
                    if u < 1.0 / (1.0 + (-(base + d[i * 2 + k])).exp()) {
                        cat += 1;
                    } else {
                        break;
                    }
                }
                y.push(cat);
            }
            groups.push(group);
        }
    }
    (y, groups)
}

#[test]
fn none_path_matches_main_reference_single_and_multigroup() {
    let (y, groups) = data();
    let map = [0, 0, 0, 1, 1, 1];
    let single = fit_bifactor_grm(
        &y[..200 * 6],
        None,
        &map,
        200,
        6,
        2,
        3,
        &BifactorGrmConfig {
            q_general: 7,
            q_specific: 7,
            max_iter: 500,
            tol: 1e-4,
            n_starts: 1,
            seed: 42,
            newton_iter: 10,
            ridge: 1e-8,
            slope_prior: SlopePrior::None,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("single-group reference fit");
    close(
        "single a_general",
        &single.a_general,
        &[
            2.0272050014392873,
            1.285739196865485,
            1.2079585618977504,
            1.9453249675111515,
            1.294463782716091,
            0.8933373440274903,
        ],
    );
    close(
        "single a_specific",
        &single.a_specific,
        &[
            0.8611996602389496,
            0.9955706255745763,
            1.1233252484727572,
            0.6417483962918548,
            1.0193487712858567,
            1.1796081512795034,
        ],
    );
    close(
        "single threshold",
        &single.threshold,
        &[
            1.3317043309056136,
            -0.8450001675222133,
            1.1387580392353476,
            -1.0936104107344924,
            1.1088742119779962,
            -0.4474721430629058,
            1.7312200278773937,
            -0.8810744823586363,
            0.8487998743620532,
            -1.1710019846389284,
            1.3283171783602505,
            -0.9718845882958843,
        ],
    );
    close(
        "single theta sums",
        &[
            single.theta_g_eap.iter().sum(),
            single.theta_g_sd.iter().sum(),
        ],
        &[2.208182722575526, 116.38435612392958],
    );
    close(
        "single loglik/change",
        &[
            *single.loglik_trace.last().unwrap(),
            single.final_loglik_change,
        ],
        &[-1199.0525539182904, 0.10103628608544568],
    );
    assert_eq!(
        single.category_counts,
        [63, 61, 76, 62, 73, 65, 64, 51, 85, 51, 75, 74, 72, 66, 62, 56, 78, 66]
    );
    assert_eq!(
        (
            single.loglik_trace.len(),
            single.n_iter,
            single.converged,
            single.best_start
        ),
        (9, 8, true, 0)
    );
    assert_eq!(single.termination_reason, "tolerance_met");
    assert_eq!(single.n_parameters, 24);

    let mg = fit_bifactor_grm_multigroup(
        &y,
        None,
        &groups,
        2,
        &map,
        400,
        6,
        2,
        3,
        Some(&[true, true, false, true, true, false]),
        &BifactorMultigroupConfig {
            q_general: 7,
            q_specific: 7,
            max_iter: 500,
            tol: 1e-4,
            n_starts: 1,
            seed: 42,
            newton_iter: 10,
            ridge: 1e-8,
            slope_prior: SlopePrior::None,
            estimate_specific_vars: false,
            device: mlsirm_core::Device::Cpu,
        },
    )
    .expect("multigroup reference fit");
    close(
        "mg a_general",
        &mg.a_general.concat(),
        &[
            1.9614830981547882,
            1.3641067535521996,
            1.1695210582863294,
            1.6316328396781958,
            1.07630643520811,
            0.8372748364953742,
            1.9614830981547882,
            1.3641067535521996,
            1.0441525498640103,
            1.6316328396781958,
            1.07630643520811,
            0.6595919612201709,
        ],
    );
    close(
        "mg a_specific",
        &mg.a_specific.concat(),
        &[
            1.1387729064861547,
            1.0299881368987944,
            1.0769372119319924,
            0.8059539879963828,
            0.8064424143167602,
            1.2925268330635544,
            1.1387729064861547,
            1.0299881368987944,
            0.9586310136008183,
            0.8059539879963828,
            0.8064424143167602,
            0.8068330922120418,
        ],
    );
    close(
        "mg threshold",
        &mg.threshold.concat(),
        &[
            1.538413428030121,
            -0.7630660759288651,
            1.3791888796052496,
            -0.8960248101111967,
            1.1943097483818612,
            -0.33533721323723903,
            1.6279404480284496,
            -0.7711256411776607,
            0.8943175688339889,
            -0.8017087381352053,
            1.4258827429768042,
            -0.9264446426390445,
            1.538413428030121,
            -0.7630660759288651,
            1.3791888796052496,
            -0.8960248101111967,
            1.7984866444913539,
            -0.13743862777917926,
            1.6279404480284496,
            -0.7711256411776607,
            0.8943175688339889,
            -0.8017087381352053,
            1.6122584240756535,
            -0.6448659688058626,
        ],
    );
    close(
        "mg population",
        &[
            &mg.general_mean[..],
            &mg.general_sd[..],
            &mg.specific_sd.concat()[..],
        ]
        .concat(),
        &[
            0.0,
            0.3292159621215796,
            1.0,
            0.9035084547905801,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    );
    close(
        "mg theta sums",
        &[mg.theta_g_eap.iter().sum(), mg.theta_g_sd.iter().sum()],
        &[58.97907318800071, 246.55962137342715],
    );
    close(
        "mg loglik/change",
        &[*mg.loglik_trace.last().unwrap(), mg.final_loglik_change],
        &[-2383.103085381413, 0.21063069717320104],
    );
    assert_eq!(
        mg.group_category_counts,
        [
            vec![63, 61, 76, 62, 73, 65, 64, 51, 85, 51, 75, 74, 72, 66, 62, 56, 78, 66],
            vec![42, 63, 95, 40, 71, 89, 32, 60, 108, 40, 69, 91, 54, 57, 89, 35, 82, 83],
        ]
    );
    assert_eq!(
        (
            mg.loglik_trace.len(),
            mg.n_iter,
            mg.converged,
            mg.best_start
        ),
        (10, 9, true, 0)
    );
    assert_eq!(mg.termination_reason, "tolerance_met");
    assert_eq!(mg.n_parameters, 34);
}
