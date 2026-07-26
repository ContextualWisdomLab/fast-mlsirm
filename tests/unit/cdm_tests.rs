use super::*;

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn bern(&mut self, p: f64) -> f64 {
        if self.next_f64() < p {
            1.0
        } else {
            0.0
        }
    }
    fn profile(&mut self, l: usize) -> usize {
        ((self.next_f64() * l as f64) as usize).min(l - 1)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[test]
fn sequential_cdm_validation_boundaries_are_explicit() {
    let base = CdmConfig::default();
    let y = [0.0, 0.0, 1.0, 1.0];
    let observed = [true; 4];
    let q = [1u8, 1];

    assert!(validate(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            init_guess: f64::NAN,
            ..base
        }
    )
    .is_err());
    assert!(validate_seq_gdina(&y, &observed, &q, 0, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &q, 2, 2, 0, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &q, 2, 2, 16, &base).is_err());
    assert!(validate_seq_gdina(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            max_iter: 0,
            ..base
        }
    )
    .is_err());
    assert!(
        validate_seq_gdina(&y, &observed, &q, 2, 2, 1, &CdmConfig { tol: 0.0, ..base }).is_err()
    );
    assert!(
        validate_seq_gdina(&y, &observed, &q, 2, 2, 1, &CdmConfig { eps: 0.0, ..base }).is_err()
    );
    assert!(validate_seq_gdina(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            init_slip: f64::NAN,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            init_guess: f64::NAN,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            init_slip: 0.6,
            init_guess: 0.6,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina(
        &y,
        &observed,
        &q,
        2,
        2,
        1,
        &CdmConfig {
            count_floor: -1.0,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina(&y[..3], &observed, &q, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &[1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&[f64::NAN, 0.0, 1.0, 1.0], &observed, &q, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &[2, 1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &[false, true, false, true], &q, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&[0.0; 4], &observed, &q, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&[65.0, 0.0, 1.0, 1.0], &observed, &q, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &[0, 1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina(&y, &observed, &[1, 0, 1, 0], 2, 2, 2, &base).is_err());

    let steps = [1usize, 1];
    let step_q = [1u8, 1];
    assert!(validate_seq_gdina_qr(&y, &observed, &step_q, &steps, 0, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &step_q, &steps, 2, 2, 0, &base).is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig {
            max_iter: 0,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig { tol: 0.0, ..base },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig { eps: 0.0, ..base },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig {
            init_slip: f64::NAN,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig {
            init_guess: f64::NAN,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig {
            init_slip: 0.6,
            init_guess: 0.6,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &CdmConfig {
            count_floor: -1.0,
            ..base
        },
    )
    .is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &step_q, &[1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &step_q, &[0, 1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &step_q, &[65, 1], 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &[1], &steps, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y[..3], &observed, &step_q, &steps, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(
        &[f64::NAN, 0.0, 1.0, 1.0],
        &observed,
        &step_q,
        &steps,
        2,
        2,
        1,
        &base,
    )
    .is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &[2, 1], &steps, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &[0, 1], &steps, 2, 2, 1, &base).is_err());
    assert!(validate_seq_gdina_qr(&y, &observed, &[1, 0, 1, 0], &steps, 2, 2, 2, &base,).is_err());
    assert!(validate_seq_gdina_qr(
        &y,
        &[false, true, false, true],
        &step_q,
        &steps,
        2,
        2,
        1,
        &base,
    )
    .is_err());
    assert!(validate_seq_gdina_qr(&[0.0; 4], &observed, &step_q, &steps, 2, 2, 1, &base).is_err());
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
}
fn bias(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n
}

fn qmask_of(q: &[u8], i: usize, k: usize) -> usize {
    let mut m = 0usize;
    for a in 0..k {
        if q[i * k + a] != 0 {
            m |= 1 << a;
        }
    }
    m
}
fn eta_of(model: CdmModel, c: usize, mask: usize) -> u8 {
    match model {
        CdmModel::Dina => ((c & mask) == mask) as u8,
        CdmModel::Dino => ((c & mask) != 0) as u8,
    }
}

/// Draw responses for the given true profiles using the same bit encoding as the estimator.
fn simulate(
    model: CdmModel,
    q: &[u8],
    s: &[f64],
    g: &[f64],
    profiles: &[usize],
    n_items: usize,
    n_attr: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let n = profiles.len();
    let mut y = vec![0.0f64; n * n_items];
    for j in 0..n {
        for i in 0..n_items {
            let mask = qmask_of(q, i, n_attr);
            let eta = eta_of(model, profiles[j], mask);
            let p = if eta == 1 { 1.0 - s[i] } else { g[i] };
            y[j * n_items + i] = rng.bern(p);
        }
    }
    y
}

fn pattern_agreement(map: &[u32], truth: &[usize]) -> f64 {
    let ok = map
        .iter()
        .zip(truth)
        .filter(|(m, t)| **m as usize == **t)
        .count();
    ok as f64 / map.len() as f64
}
fn attribute_agreement(attr_prob: &[f64], truth: &[usize], n: usize, k: usize) -> f64 {
    let mut ok = 0usize;
    for j in 0..n {
        for a in 0..k {
            let est = (attr_prob[j * k + a] >= 0.5) as usize;
            let tru = (truth[j] >> a) & 1;
            if est == tru {
                ok += 1;
            }
        }
    }
    ok as f64 / (n * k) as f64
}
fn nondecreasing(trace: &[f64]) -> bool {
    trace.windows(2).all(|w| w[1] >= w[0] - 1e-6)
}
fn monotone_items(res: &CdmResult) -> bool {
    // 1 - s_i > g_i, with slack for the extreme clamp corner (1-s = g = eps).
    res.slip
        .iter()
        .zip(&res.guess)
        .all(|(s, g)| 1.0 - s > g - 1e-9)
}

/// Anchor 1: the eta bitmask + likelihood algebra, with zero estimation. `P(X_j)`
/// from the module's log-space path must equal a naive enumeration that expands
/// `eta = prod_k alpha^{q}` in plain arithmetic.
#[test]
fn anchor_brute_force_likelihood() {
    let (n_attr, n_items, l) = (2usize, 2usize, 4usize);
    let q: Vec<u8> = vec![1, 0, /* */ 1, 1];
    let s = [0.1f64, 0.2];
    let g = [0.15f64, 0.2];
    let pi = [0.4f64, 0.2, 0.1, 0.3];
    let x = [1.0f64, 0.0];
    let model = CdmModel::Dina;

    let mut eta = vec![0u8; n_items * l];
    let mut lp1 = vec![0.0f64; n_items * 2];
    let mut lp0 = vec![0.0f64; n_items * 2];
    for i in 0..n_items {
        let mask = qmask_of(&q, i, n_attr);
        for c in 0..l {
            eta[i * l + c] = eta_of(model, c, mask);
        }
        lp1[i * 2 + 1] = (1.0 - s[i]).ln();
        lp0[i * 2 + 1] = s[i].ln();
        lp1[i * 2] = g[i].ln();
        lp0[i * 2] = (1.0 - g[i]).ln();
    }
    let log_pi: Vec<f64> = pi.iter().map(|p| p.ln()).collect();
    let observed = vec![true; n_items];
    let mut post = vec![0.0f64; l];
    let log_px = posterior_row(
        0, &x, &observed, n_items, l, &eta, &lp1, &lp0, &log_pi, &mut post,
    );

    let mut px = 0.0;
    for c in 0..l {
        let mut lik = pi[c];
        for i in 0..n_items {
            let mut e = 1u8;
            for k in 0..n_attr {
                if q[i * n_attr + k] == 1 {
                    e *= ((c >> k) & 1) as u8; // AND gate as a product
                }
            }
            let pc = if e == 1 { 1.0 - s[i] } else { g[i] };
            let xi = x[i];
            lik *= pc.powf(xi) * (1.0 - pc).powf(1.0 - xi);
        }
        px += lik;
    }
    assert!(
        (log_px.exp() - px).abs() < 1e-12,
        "module {} vs naive {}",
        log_px.exp(),
        px
    );
    assert!((post.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

/// Anchor 2: deterministic limit s=g=0 => X = eta exactly. Recovery of the ideal
/// pattern must be perfect and recovered slip/guess near zero.
#[test]
fn anchor_deterministic_limit() {
    let (n_attr, n_items) = (2usize, 3usize);
    let q: Vec<u8> = vec![1, 0, /* */ 0, 1, /* */ 1, 1];
    let s = vec![0.0f64; n_items];
    let g = vec![0.0f64; n_items];
    let n = 400usize;
    let profiles: Vec<usize> = (0..n).map(|j| j % 4).collect();
    let mut rng = Lcg(12345);
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged);
    assert!(nondecreasing(&res.loglik_trace));
    assert!(monotone_items(&res));
    assert!(pattern_agreement(&res.map_profile, &profiles) > 0.99);
    assert!(res.slip.iter().all(|&s| s < 1e-2), "slip {:?}", res.slip);
    assert!(res.guess.iter().all(|&g| g < 1e-2), "guess {:?}", res.guess);
}

/// Anchor 3: with a single-attribute-per-item Q, `(c & mask) == mask` and
/// `(c & mask) != 0` coincide, so DINA and DINO share bit-identical eta and, from
/// the deterministic init, must produce identical fits. Pure algebraic identity.
#[test]
fn anchor_dina_dino_gate_identity() {
    let (n_attr, n_items) = (2usize, 4usize);
    let q: Vec<u8> = vec![1, 0, /* */ 1, 0, /* */ 0, 1, /* */ 0, 1];
    let s = vec![0.15f64; n_items];
    let g = vec![0.2f64; n_items];
    let n = 500usize;
    let mut rng = Lcg(999);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let cfg = CdmConfig::default();
    let a = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg).unwrap();
    let b = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dino, &cfg).unwrap();
    assert!(rmse(&a.slip, &b.slip) < 1e-9);
    assert!(rmse(&a.guess, &b.guess) < 1e-9);
    assert!(rmse(&a.profile_prob, &b.profile_prob) < 1e-9);
}

/// Anchor 4: K=1, Q all-ones reduces to a 2-class latent-class model. Recover the
/// master proportion, slip and guess.
#[test]
fn anchor_k1_two_class_reduction() {
    let (n_attr, n_items) = (1usize, 10usize);
    let q: Vec<u8> = vec![1u8; n_items];
    let (s_true, g_true, pi1) = (0.15f64, 0.2f64, 0.6f64);
    let s = vec![s_true; n_items];
    let g = vec![g_true; n_items];
    let n = 2000usize;
    let mut rng = Lcg(7);
    let profiles: Vec<usize> = (0..n)
        .map(|_| if rng.next_f64() < pi1 { 1 } else { 0 })
        .collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged && monotone_items(&res));
    let mean_s = res.slip.iter().sum::<f64>() / n_items as f64;
    let mean_g = res.guess.iter().sum::<f64>() / n_items as f64;
    assert!((mean_s - s_true).abs() < 0.05, "mean slip {mean_s}");
    assert!((mean_g - g_true).abs() < 0.05, "mean guess {mean_g}");
    assert!(
        (res.profile_prob[1] - pi1).abs() < 0.05,
        "pi1 {}",
        res.profile_prob[1]
    );
}

/// Tier-1 fast recovery guard: K=2, J=15, N=1000, s=g=0.2, identifiable Q.
#[test]
fn recovery_guard() {
    let (n_attr, n_items, n) = (2usize, 15usize, 1000usize);
    // 5 items {a0}, 5 items {a1}, 5 items {a0,a1}.
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..15 {
        if i < 5 {
            q[i * 2] = 1;
        } else if i < 10 {
            q[i * 2 + 1] = 1;
        } else {
            q[i * 2] = 1;
            q[i * 2 + 1] = 1;
        }
    }
    let s = vec![0.2f64; n_items];
    let g = vec![0.2f64; n_items];
    let mut rng = Lcg(2024);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged);
    assert!(nondecreasing(&res.loglik_trace));
    assert!(monotone_items(&res));
    assert!(
        rmse(&res.slip, &s) < 0.05,
        "rmse slip {}",
        rmse(&res.slip, &s)
    );
    assert!(
        rmse(&res.guess, &g) < 0.05,
        "rmse guess {}",
        rmse(&res.guess, &g)
    );
    assert!(pattern_agreement(&res.map_profile, &profiles) > 0.80);
    assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85);
    assert_eq!(res.n_parameters, 2 * n_items + ((1 << n_attr) - 1));
}

/// Missing-data (MAR) path: masked cells are dropped from likelihood and counts.
#[test]
fn handles_missing_data() {
    let (n_attr, n_items, n) = (2usize, 8usize, 400usize);
    let q: Vec<u8> = vec![
        1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1, /* */ 1, 1,
        /* */ 1, 0, /* */ 0, 1,
    ];
    let s = vec![0.15f64; n_items];
    let g = vec![0.2f64; n_items];
    let mut rng = Lcg(555);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let mut observed = vec![true; n * n_items];
    for (idx, o) in observed.iter_mut().enumerate() {
        if rng.next_f64() < 0.2 {
            *o = false; // ~20% MCAR missing
        }
        let _ = idx;
    }
    let res = fit_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged && monotone_items(&res));
    assert!(nondecreasing(&res.loglik_trace));
}

/// Directly exercise every M-step branch (normal, both count guards, projection).
#[test]
fn update_item_branches() {
    let cfg = CdmConfig::default();
    let mut s = vec![0.2, 0.2, 0.2, 0.2];
    let mut g = vec![0.2, 0.2, 0.2, 0.2];
    // 0: normal — masters mostly right, non-masters mostly wrong.
    // 1: I1 below floor -> keep previous slip.
    // 2: I0 below floor -> keep previous guess.
    // 3: monotonicity violation (masters worse than non-masters) -> projection.
    let i1 = vec![100.0, 1e-12, 100.0, 100.0];
    let r1 = vec![80.0, 0.0, 80.0, 20.0];
    let i0 = vec![100.0, 100.0, 1e-12, 100.0];
    let r0 = vec![20.0, 20.0, 0.0, 80.0];
    for i in 0..4 {
        update_item(i, &i1, &r1, &i0, &r0, &mut s, &mut g, &cfg);
    }
    assert!((s[0] - 0.2).abs() < 1e-9 && (g[0] - 0.2).abs() < 1e-9);
    assert!((s[1] - 0.2).abs() < 1e-9, "kept prev slip {}", s[1]); // guard held slip
    assert!((g[2] - 0.2).abs() < 1e-9, "kept prev guess {}", g[2]); // guard held guess
    assert!(
        1.0 - s[3] > g[3],
        "projection kept monotonicity: 1-s={} g={}",
        1.0 - s[3],
        g[3]
    );
}

/// The non-converged exit path (max_iter reached without meeting tol).
#[test]
fn stops_at_max_iter() {
    let (n_attr, n_items, n) = (1usize, 4usize, 50usize);
    let q = vec![1u8; n_items];
    let s = vec![0.1f64; n_items];
    let g = vec![0.2f64; n_items];
    let mut rng = Lcg(3);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(2)).collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let cfg = CdmConfig {
        max_iter: 1,
        ..CdmConfig::default()
    };
    let res = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg).unwrap();
    assert!(!res.converged);
    assert_eq!(res.n_iter, 1);
    assert_eq!(res.loglik_trace.len(), 2);
    assert!(nondecreasing(&res.loglik_trace));
}

/// Malformed inputs are rejected with `Err` (covers each validate branch).
#[test]
fn validate_rejects_malformed() {
    let q_ok = vec![1u8, 0, 0, 1];
    let y = vec![0.0f64; 2 * 2];
    let obs = vec![true; 4];
    let cfg = CdmConfig::default();
    let bad = |q: &[u8], y: &[f64], obs: &[bool], n: usize, j: usize, k: usize| {
        fit_cdm(y, obs, q, n, j, k, CdmModel::Dina, &cfg).is_err()
    };
    assert!(bad(&q_ok, &y, &obs, 0, 2, 2)); // n_persons < 1
    assert!(bad(&q_ok, &y, &obs, 2, 2, 0)); // K < 1
    assert!(bad(
        &vec![1u8; 2 * 16],
        &vec![0.0; 2 * 2],
        &vec![true; 4],
        2,
        2,
        16
    )); // K > 15
    assert!(bad(&q_ok, &vec![0.0; 3], &obs, 2, 2, 2)); // y length
    assert!(bad(&q_ok, &y, &vec![true; 3], 2, 2, 2)); // observed length
    assert!(bad(&vec![1u8; 3], &y, &obs, 2, 2, 2)); // q length
    assert!(bad(&q_ok, &vec![2.0, 0.0, 0.0, 0.0], &obs, 2, 2, 2)); // y not in {0,1}
    assert!(bad(&vec![2u8, 0, 0, 1], &y, &obs, 2, 2, 2)); // q not in {0,1}
    assert!(bad(&vec![0u8, 0, 1, 1], &y, &obs, 2, 2, 2)); // all-zero Q row 0
    assert!(bad(&vec![1u8, 0, 1, 0], &y, &obs, 2, 2, 2)); // all-zero Q column 1
                                                          // Item 1 is entirely missing, so its slip/guess cannot be estimated.
    assert!(bad(&q_ok, &y, &[true, false, true, false], 2, 2, 2));
    // A well-formed call still succeeds.
    assert!(fit_cdm(&y, &obs, &q_ok, 2, 2, 2, CdmModel::Dina, &cfg).is_ok());
}

#[test]
fn validate_rejects_invalid_config() {
    let q = vec![1u8, 0, 0, 1];
    let y = vec![0.0f64; 4];
    let observed = vec![true; 4];
    let rejected =
        |cfg: CdmConfig| fit_cdm(&y, &observed, &q, 2, 2, 2, CdmModel::Dina, &cfg).is_err();
    assert!(rejected(CdmConfig {
        max_iter: 0,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        tol: f64::NAN,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        eps: 0.5,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        eps: 1e-3,
        mono_backoff: 2e-3,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        init_slip: f64::INFINITY,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        init_slip: 0.6,
        init_guess: 0.4,
        ..CdmConfig::default()
    }));
    assert!(rejected(CdmConfig {
        count_floor: -1.0,
        ..CdmConfig::default()
    }));
}

/// Literature-grade Monte-Carlo (>=500 reps): de la Torre (2009)-style design,
/// recovering slip/guess (RMSE/bias) and attribute/pattern classification accuracy.
/// Q is held to moderate complexity (1-2 attribute items) so the aggregate RMSE
/// bound holds (a 3-attribute item shrinks the eta=1 group to ~N/8 and inflates SE).
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_cdm_recovery() {
    let (n_attr, n_items, n, reps) = (5usize, 30usize, 1000usize, 500usize);
    let l = 1usize << n_attr;
    // 20 single-attribute items (4 per attribute) + 10 two-attribute items (pairs).
    let mut q = vec![0u8; n_items * n_attr];
    for a in 0..5 {
        for r in 0..4 {
            q[(a * 4 + r) * n_attr + a] = 1;
        }
    }
    let pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 3),
        (1, 4),
        (0, 4),
    ];
    for (t, &(a, b)) in pairs.iter().enumerate() {
        q[(20 + t) * n_attr + a] = 1;
        q[(20 + t) * n_attr + b] = 1;
    }

    for (cond, &sg) in [0.1f64, 0.2].iter().enumerate() {
        let s_true = vec![sg; n_items];
        let g_true = vec![sg; n_items];
        let (mut sum_rs, mut sum_rg, mut sum_bs, mut sum_bg) = (0.0, 0.0, 0.0, 0.0);
        let (mut ss_rs, mut ss_rg) = (0.0, 0.0);
        let (mut sum_pat, mut sum_attr) = (0.0, 0.0);
        for rep in 0..reps {
            let seed = 0xD1B54A32D192ED03u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((cond as u64 + 1) * 0x9E3779B97F4A7C15);
            let mut rng = Lcg(seed);
            let profiles: Vec<usize> = (0..n).map(|_| rng.profile(l)).collect();
            let y = simulate(
                CdmModel::Dina,
                &q,
                &s_true,
                &g_true,
                &profiles,
                n_items,
                n_attr,
                &mut rng,
            );
            let observed = vec![true; n * n_items];
            let res = fit_cdm(
                &y,
                &observed,
                &q,
                n,
                n_items,
                n_attr,
                CdmModel::Dina,
                &CdmConfig::default(),
            )
            .unwrap();
            let (rs, rg) = (rmse(&res.slip, &s_true), rmse(&res.guess, &g_true));
            sum_rs += rs;
            sum_rg += rg;
            ss_rs += rs * rs;
            ss_rg += rg * rg;
            sum_bs += bias(&res.slip, &s_true);
            sum_bg += bias(&res.guess, &g_true);
            sum_pat += pattern_agreement(&res.map_profile, &profiles);
            sum_attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
        }
        let r = reps as f64;
        let (m_rs, m_rg) = (sum_rs / r, sum_rg / r);
        let sd_rs = (ss_rs / r - m_rs * m_rs).max(0.0).sqrt();
        let sd_rg = (ss_rg / r - m_rg * m_rg).max(0.0).sqrt();
        println!(
            "s=g={:.1}: RMSE(s)={:.4}(SD {:.4}) RMSE(g)={:.4}(SD {:.4}) bias(s)={:.4} bias(g)={:.4} pattern={:.3} attribute={:.3}",
            sg, m_rs, sd_rs, m_rg, sd_rg, sum_bs / r, sum_bg / r, sum_pat / r, sum_attr / r
        );
        assert!(m_rs < 0.03, "mean RMSE(s) {m_rs} at s=g={sg}");
        assert!(m_rg < 0.03, "mean RMSE(g) {m_rg} at s=g={sg}");
        if sg == 0.1 {
            assert!(
                sum_attr / r > 0.90,
                "mean attribute agreement {} at s=g=0.1",
                sum_attr / r
            );
        }
    }
}

// ----- G-DINA (saturated) tests -----

/// Build the ragged CSR layout (item_off, qmask, k_required) from a Q-matrix,
/// matching fit_gdina exactly.
fn gdina_layout(q: &[u8], n_items: usize, n_attr: usize) -> (Vec<usize>, Vec<usize>, Vec<u32>) {
    let mut qmask = vec![0usize; n_items];
    let mut kreq = vec![0u32; n_items];
    for i in 0..n_items {
        let m = qmask_of(q, i, n_attr);
        qmask[i] = m;
        kreq[i] = m.count_ones();
    }
    let mut off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        off[i + 1] = off[i] + (1usize << kreq[i]);
    }
    (off, qmask, kreq)
}

/// Draw responses from a CSR-flat truth table, using the SAME reduce_class + item_off
/// convention as the estimator so RMSE compares matched classes (spec fix 3).
fn simulate_gdina(
    qmask: &[usize],
    item_off: &[usize],
    truth_p: &[f64],
    profiles: &[usize],
    n_items: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let n = profiles.len();
    let mut y = vec![0.0f64; n * n_items];
    for j in 0..n {
        for i in 0..n_items {
            let l = reduce_class(profiles[j], qmask[i]);
            y[j * n_items + i] = rng.bern(truth_p[item_off[i] + l]);
        }
    }
    y
}

/// Check the all-mastered class for monotone-truth fixtures only; this is not an
/// invariant of the unconstrained saturated G-DINA estimator.
fn top_class_is_max(res: &GdinaResult) -> bool {
    (0..res.k_required.len()).all(|i| {
        let (a, b) = (res.item_off[i], res.item_off[i + 1]);
        let top = res.item_prob[b - 1];
        res.item_prob[a..b].iter().all(|&p| p <= top + 1e-9)
    })
}

/// reduce_class packs the required-attribute mastery bits LSB-ascending, and
/// equals L_i-1 iff all required attributes are mastered (the DINA eta identity).
#[test]
fn gdina_reduce_class_matches_bruteforce() {
    for k in 1..=4usize {
        for qmask in 1..(1usize << k) {
            let li = 1usize << (qmask.count_ones());
            for c in 0..(1usize << k) {
                let (mut expect, mut m) = (0usize, 0u32);
                for bit in 0..k {
                    if (qmask >> bit) & 1 == 1 {
                        expect |= ((c >> bit) & 1) << m;
                        m += 1;
                    }
                }
                assert_eq!(reduce_class(c, qmask), expect);
                assert_eq!(reduce_class(c, qmask) == li - 1, (c & qmask) == qmask);
            }
        }
    }
}

/// mobius_inverse_inplace is the exact inverse of the zeta subset-sum, and matches
/// the explicit K=2 identity-link formulas.
#[test]
fn gdina_mobius_roundtrip() {
    let mut rng = Lcg(42);
    for ki in 1..=3u32 {
        let li = 1usize << ki;
        let p: Vec<f64> = (0..li).map(|_| 0.05 + 0.9 * rng.next_f64()).collect();
        let mut delta = p.clone();
        mobius_inverse_inplace(&mut delta, ki);
        for l in 0..li {
            // reconstruct p_l = sum_{S subset of l} delta_S
            let recon: f64 = (0..li).filter(|&s| (l & s) == s).map(|s| delta[s]).sum();
            assert!((recon - p[l]).abs() < 1e-12, "roundtrip K={ki} l={l}");
        }
    }
    let mut d = vec![0.2, 0.5, 0.6, 0.9]; // p00, p10, p01, p11
    mobius_inverse_inplace(&mut d, 2);
    assert!((d[0] - 0.2).abs() < 1e-12);
    assert!((d[1] - (0.5 - 0.2)).abs() < 1e-12);
    assert!((d[2] - (0.6 - 0.2)).abs() < 1e-12);
    assert!((d[3] - (0.9 - 0.5 - 0.6 + 0.2)).abs() < 1e-12);
}

/// Brute-force likelihood: the CSR log-space path equals a naive enumeration.
#[test]
fn gdina_brute_force_likelihood() {
    let (n_attr, n_items) = (2usize, 2usize);
    let l_full = 1usize << n_attr;
    let q: Vec<u8> = vec![1, 0, /* */ 1, 1]; // item 0: K=1, item 1: K=2
    let (item_off, qmask, _k) = gdina_layout(&q, n_items, n_attr);
    let total = item_off[n_items];
    let p = vec![0.15f64, 0.8, /* */ 0.1, 0.3, 0.4, 0.85];
    assert_eq!(p.len(), total);
    let mut red = vec![0u16; n_items * l_full];
    for i in 0..n_items {
        for c in 0..l_full {
            red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
        }
    }
    let (mut log_p1, mut log_p0) = (vec![0.0f64; total], vec![0.0f64; total]);
    for x in 0..total {
        log_p1[x] = p[x].ln();
        log_p0[x] = (1.0 - p[x]).ln();
    }
    let pi = [0.4f64, 0.2, 0.1, 0.3];
    let log_pi: Vec<f64> = pi.iter().map(|v| v.ln()).collect();
    let x = [1.0f64, 0.0];
    let observed = vec![true; n_items];
    let mut post = vec![0.0f64; l_full];
    let log_px = posterior_row_gdina(
        0, &x, &observed, n_items, l_full, &red, &log_p1, &log_p0, &item_off, &log_pi, &mut post,
    );
    let mut px = 0.0;
    for c in 0..l_full {
        let mut lik = pi[c];
        for i in 0..n_items {
            let pc = p[item_off[i] + reduce_class(c, qmask[i])];
            let xi = x[i];
            lik *= pc.powf(xi) * (1.0 - pc).powf(1.0 - xi);
        }
        px += lik;
    }
    assert!(
        (log_px.exp() - px).abs() < 1e-12,
        "module {} vs naive {}",
        log_px.exp(),
        px
    );
    assert!((post.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

/// THE CRUX ANCHOR: DINA-generated data => the saturated fit recovers p = g for
/// every non-top reduced class and 1-s at the top, so delta has only the intercept
/// and the highest-order interaction nonzero (the exact DINA identity-link constraint).
#[test]
fn gdina_recovers_dina() {
    let (n_attr, n_items, n) = (2usize, 12usize, 2500usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..n_items {
        if i < 4 {
            q[i * 2] = 1;
        } else if i < 8 {
            q[i * 2 + 1] = 1;
        } else {
            q[i * 2] = 1;
            q[i * 2 + 1] = 1;
        }
    }
    let s = vec![0.15f64; n_items];
    let g = vec![0.2f64; n_items];
    let mut rng = Lcg(2011);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace) && top_class_is_max(&res));
    let (item_off, _qm, _k) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a, b) = (item_off[i], item_off[i + 1]);
        for l in a..b {
            truth[l] = g[i];
        }
        truth[b - 1] = 1.0 - s[i];
    }
    assert!(
        rmse(&res.item_prob, &truth) < 0.03,
        "DINA p RMSE {}",
        rmse(&res.item_prob, &truth)
    );
    for i in 0..n_items {
        let (a, b) = (item_off[i], item_off[i + 1]);
        let d = &res.item_delta[a..b];
        assert!((d[0] - g[i]).abs() < 0.05, "delta0 {} vs g {}", d[0], g[i]);
        assert!(
            (d[b - a - 1] - ((1.0 - s[i]) - g[i])).abs() < 0.05,
            "delta_full item {i}"
        );
        for l in 1..(b - a - 1) {
            assert!(
                d[l].abs() < 0.05,
                "interior delta item {i} idx {l} = {}",
                d[l]
            );
        }
    }
}

/// DINO-generated data: p = g at the empty reduced class, 1-s elsewhere. Uses a
/// mixed Q (single-attribute items identify the attributes; an all-two-attribute Q
/// would leave profiles 10/01/11 response-equivalent under the OR gate).
#[test]
fn gdina_recovers_dino() {
    let (n_attr, n_items, n) = (2usize, 12usize, 2500usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..n_items {
        if i < 4 {
            q[i * 2] = 1;
        } else if i < 8 {
            q[i * 2 + 1] = 1;
        } else {
            q[i * 2] = 1;
            q[i * 2 + 1] = 1;
        }
    }
    let s = vec![0.15f64; n_items];
    let g = vec![0.2f64; n_items];
    let mut rng = Lcg(77);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate(
        CdmModel::Dino,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    let (item_off, _qm, _k) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a, b) = (item_off[i], item_off[i + 1]);
        for l in a..b {
            truth[l] = 1.0 - s[i];
        }
        truth[a] = g[i];
    }
    assert!(
        rmse(&res.item_prob, &truth) < 0.03,
        "DINO p RMSE {}",
        rmse(&res.item_prob, &truth)
    );
}

/// A-CDM (additive) data: recover p and confirm the interaction delta is ~0.
#[test]
fn gdina_recovers_acdm() {
    let (n_attr, n_items, n) = (2usize, 10usize, 4000usize);
    let q = vec![1u8; n_items * n_attr];
    let base = [0.1f64, 0.35, 0.4, 0.65]; // additive: p11 = 0.1 + 0.25 + 0.3, no interaction
    let (item_off, qmask, _k) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        for l in 0..4 {
            truth[item_off[i] + l] = base[l];
        }
    }
    let mut rng = Lcg(303);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(
        rmse(&res.item_prob, &truth) < 0.05,
        "A-CDM p RMSE {}",
        rmse(&res.item_prob, &truth)
    );
    // Additive truth => interaction terms are negligible RELATIVE to the main
    // effects (an interaction is a 4-probability contrast, so its absolute noise
    // (~0.05) makes a fixed bound flaky; the additivity claim is a small ratio).
    let (mut sum_int, mut sum_main) = (0.0, 0.0);
    for i in 0..n_items {
        let base = item_off[i];
        sum_int += res.item_delta[base + 3].abs(); // both-attribute interaction
        sum_main += (res.item_delta[base + 1].abs() + res.item_delta[base + 2].abs()) / 2.0;
    }
    assert!(
        sum_int / sum_main < 0.35,
        "A-CDM interaction/main ratio {}",
        sum_int / sum_main
    );
    assert!(top_class_is_max(&res));
}

/// Deterministic s=g=0 limit: ideal responses => exact pattern recovery.
#[test]
fn gdina_deterministic_limit() {
    let (n_attr, n_items, n) = (2usize, 3usize, 400usize);
    let q: Vec<u8> = vec![1, 0, /* */ 0, 1, /* */ 1, 1];
    let s = vec![0.0f64; n_items];
    let g = vec![0.0f64; n_items];
    let profiles: Vec<usize> = (0..n).map(|j| j % 4).collect();
    let mut rng = Lcg(9);
    let y = simulate(
        CdmModel::Dina,
        &q,
        &s,
        &g,
        &profiles,
        n_items,
        n_attr,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.converged && top_class_is_max(&res));
    assert!(pattern_agreement(&res.map_profile, &profiles) > 0.99);
}

/// Tier-1 fast recovery guard: K=2, J=15, N=1000, monotone saturated truth.
#[test]
fn gdina_recovery_guard() {
    let (n_attr, n_items, n) = (2usize, 15usize, 1000usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..15 {
        if i < 5 {
            q[i * 2] = 1;
        } else if i < 10 {
            q[i * 2 + 1] = 1;
        } else {
            q[i * 2] = 1;
            q[i * 2 + 1] = 1;
        }
    }
    let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let a = item_off[i];
        if kreq[i] == 1 {
            truth[a] = 0.2;
            truth[a + 1] = 0.8;
        } else {
            truth[a] = 0.2;
            truth[a + 1] = 0.5;
            truth[a + 2] = 0.55;
            truth[a + 3] = 0.85;
        }
    }
    let mut rng = Lcg(2024);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace));
    assert!(
        rmse(&res.item_prob, &truth) < 0.05,
        "guard p RMSE {}",
        rmse(&res.item_prob, &truth)
    );
    assert!(top_class_is_max(&res));
    assert!(pattern_agreement(&res.map_profile, &profiles) > 0.80);
    assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85);
    let total: usize = (0..n_items).map(|i| 1usize << kreq[i]).sum();
    assert_eq!(res.n_parameters, total + ((1 << n_attr) - 1));
}

/// Missing-at-random cells are dropped from both likelihood and reduced-class counts.
#[test]
fn gdina_handles_missing_data() {
    let (n_attr, n_items, n) = (2usize, 9usize, 500usize);
    let q: Vec<u8> = vec![
        1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1, /* */ 1, 1,
        /* */ 1, 0, /* */ 0, 1, /* */ 1, 1,
    ];
    let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let a = item_off[i];
        if kreq[i] == 1 {
            truth[a] = 0.2;
            truth[a + 1] = 0.8;
        } else {
            truth[a] = 0.15;
            truth[a + 1] = 0.5;
            truth[a + 2] = 0.55;
            truth[a + 3] = 0.85;
        }
    }
    let mut rng = Lcg(555);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let mut observed = vec![true; n * n_items];
    for o in observed.iter_mut() {
        if rng.next_f64() < 0.2 {
            *o = false;
        }
    }
    let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.converged && top_class_is_max(&res));
    assert!(nondecreasing(&res.loglik_trace));
}

/// Literature-grade Monte-Carlo (>=500 reps): de la Torre (2011)-style design.
/// Attributes are drawn from a STOCHASTIC higher-order logistic model (de la Torre
/// & Douglas, 2004) so every reduced class gets positive, correlated mass; RMSE(p)
/// is mass-weighted so near-empty classes don't dominate (spec fixes 1 & 2). Q is
/// held to 1-2 required attributes per item to keep the reduced classes populated.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_gdina_recovery() {
    let (n_attr, n_items, n, reps) = (5usize, 30usize, 1000usize, 500usize);
    let mut q = vec![0u8; n_items * n_attr];
    for a in 0..5 {
        for r in 0..4 {
            q[(a * 4 + r) * n_attr + a] = 1;
        }
    }
    let pairs = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 2),
        (1, 3),
        (2, 4),
        (0, 3),
        (1, 4),
        (0, 4),
    ];
    for (t, &(a, b)) in pairs.iter().enumerate() {
        q[(20 + t) * n_attr + a] = 1;
        q[(20 + t) * n_attr + b] = 1;
    }
    let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
    let total = item_off[n_items];
    let bk = [-1.0f64, -0.5, 0.0, 0.5, 1.0];
    let lambda = 1.5f64;

    for &skew in [false, true].iter() {
        for &sg in [0.1f64, 0.2].iter() {
            // Additive monotone truth: p_il = sg + (1-2sg)*popcount(l)/K_i.
            let mut truth = vec![0.0f64; total];
            for i in 0..n_items {
                let ki = kreq[i] as f64;
                for l in 0..(item_off[i + 1] - item_off[i]) {
                    truth[item_off[i] + l] = sg + (1.0 - 2.0 * sg) * (l.count_ones() as f64) / ki;
                }
            }
            let mut dtruth = truth.clone();
            for i in 0..n_items {
                mobius_inverse_inplace(&mut dtruth[item_off[i]..item_off[i + 1]], kreq[i]);
            }
            let (mut sum_wp, mut sum_bp, mut sum_dp, mut sum_pat, mut sum_attr) =
                (0.0, 0.0, 0.0, 0.0, 0.0);
            for rep in 0..reps {
                let seed = 0xD1B54A32D192ED03u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 * 2 + (sg == 0.1) as u64 + 1) * 0x9E3779B97F4A7C15);
                let mut rng = Lcg(seed);
                let profiles: Vec<usize> = (0..n)
                    .map(|_| {
                        let theta = if skew {
                            -(rng.next_f64().max(1e-12)).ln() - 1.0
                        } else {
                            rng.normal()
                        };
                        let mut c = 0usize;
                        for k in 0..n_attr {
                            let pk = 1.0 / (1.0 + (-lambda * (theta - bk[k])).exp());
                            if rng.next_f64() < pk {
                                c |= 1 << k;
                            }
                        }
                        c
                    })
                    .collect();
                let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
                let observed = vec![true; n * n_items];
                let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default())
                    .unwrap();
                // mass-weighted RMSE(p): weight each class by realized frequency.
                let mut mass = vec![0.0f64; total];
                for &c in &profiles {
                    for i in 0..n_items {
                        mass[item_off[i] + reduce_class(c, qmask[i])] += 1.0;
                    }
                }
                let (mut num, mut den) = (0.0, 0.0);
                for x in 0..total {
                    let e = res.item_prob[x] - truth[x];
                    num += mass[x] * e * e;
                    den += mass[x];
                }
                sum_wp += (num / den).sqrt();
                sum_bp += bias(&res.item_prob, &truth);
                sum_dp += rmse(&res.item_delta, &dtruth);
                sum_pat += pattern_agreement(&res.map_profile, &profiles);
                sum_attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
            }
            let r = reps as f64;
            println!(
                "skew={} s=g={:.1}: wRMSE(p)={:.4} bias(p)={:.4} RMSE(delta)={:.4} pattern={:.3} attribute={:.3}",
                skew, sg, sum_wp / r, sum_bp / r, sum_dp / r, sum_pat / r, sum_attr / r
            );
            assert!(
                sum_wp / r < 0.03,
                "mass-weighted RMSE(p) {} skew={skew} sg={sg}",
                sum_wp / r
            );
            if sg == 0.1 {
                assert!(
                    sum_attr / r > 0.90,
                    "attribute agreement {} skew={skew}",
                    sum_attr / r
                );
            }
        }
    }
}

// ----- Q-matrix validation (de la Torre & Chiu, 2016) tests -----

/// A canonical K=3, 15-item Q-matrix: six single-attribute items (two per
/// attribute), six two-attribute items (two per pair), three full-triple items.
fn canonical_q3() -> Vec<u8> {
    let k = 3usize;
    let mut q = vec![0u8; 15 * k];
    let set = |q: &mut [u8], i: usize, attrs: &[usize]| {
        for &a in attrs {
            q[i * k + a] = 1;
        }
    };
    let rows: [&[usize]; 15] = [
        &[0],
        &[1],
        &[2],
        &[0],
        &[1],
        &[2], // singles
        &[0, 1],
        &[0, 2],
        &[1, 2],
        &[0, 1],
        &[0, 2],
        &[1, 2], // pairs
        &[0, 1, 2],
        &[0, 1, 2],
        &[0, 1, 2], // triples
    ];
    for (i, r) in rows.iter().enumerate() {
        set(&mut q, i, r);
    }
    q
}

fn q_rows_equal(a: &[u8], b: &[u8], i: usize, k: usize) -> bool {
    (0..k).all(|c| (a[i * k + c] != 0) == (b[i * k + c] != 0))
}

/// ANCHOR: DINA-generated data whose provisional Q is the TRUE Q must validate
/// to itself — every item's true q-vector is the fewest-attribute vector whose
/// PVAF clears the cutoff, so nothing is flagged.
#[test]
fn qval_true_q_validates_to_itself() {
    let (k, n_items, n) = (3usize, 15usize, 3000usize);
    let q = canonical_q3();
    let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
    let mut rng = Lcg(20240715);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, k, &mut rng);
    let observed = vec![true; n * n_items];
    let res = validate_q_matrix(
        &y,
        &observed,
        &q,
        n,
        n_items,
        k,
        0.95,
        &CdmConfig::default(),
    )
    .unwrap();
    let correct = (0..n_items)
        .filter(|&i| q_rows_equal(&res.suggested_q, &q, i, k))
        .count();
    assert!(
        correct >= n_items - 1,
        "recovered {correct}/{n_items} true q-vectors"
    );
    // The true q-vector explains ~all the item variance.
    assert!(
        res.provisional_pvaf.iter().all(|&p| p > 0.9),
        "min provisional PVAF {}",
        res.provisional_pvaf
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    );
}

/// A provisional Q with BOTH under-specified pairs (one attribute dropped) and
/// over-specified singles (one spurious attribute added) is corrected back to
/// the truth, and exactly the mis-specified items are flagged.
#[test]
fn qval_corrects_over_and_under_specification() {
    let (k, n_items, n) = (3usize, 15usize, 4000usize);
    let truth = canonical_q3();
    let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
    let mut rng = Lcg(13579);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate(
        CdmModel::Dina,
        &truth,
        &s,
        &g,
        &profiles,
        n_items,
        k,
        &mut rng,
    );
    let observed = vec![true; n * n_items];

    // Mis-specify a FEW items only (the method needs the rest of the Q to keep
    // the attributes identified): over-specify singles 0 & 3, under-specify
    // pairs 6 & 9.
    let mut prov = truth.clone();
    prov[0 * k + 1] = 1; // item 0 {0} -> {0,1}
    prov[3 * k + 2] = 1; // item 3 {0} -> {0,2}
    prov[6 * k + 1] = 0; // item 6 {0,1} -> {0}
    prov[9 * k + 0] = 0; // item 9 {0,1} -> {1}
    let perturbed = [0usize, 3, 6, 9];

    let res = validate_q_matrix(
        &y,
        &observed,
        &prov,
        n,
        n_items,
        k,
        0.95,
        &CdmConfig::default(),
    )
    .unwrap();
    let correct = (0..n_items)
        .filter(|&i| q_rows_equal(&res.suggested_q, &truth, i, k))
        .count();
    assert!(
        correct >= n_items - 1,
        "corrected {correct}/{n_items} to truth"
    );
    for &i in &perturbed {
        assert!(res.flagged[i], "item {i} was mis-specified but not flagged");
        assert!(
            q_rows_equal(&res.suggested_q, &truth, i, k),
            "item {i} not corrected back to truth"
        );
    }
}

#[test]
fn qval_rejects_malformed() {
    let n = 4usize;
    let y = vec![0.0f64; n * 3];
    let obs = vec![true; n * 3];
    let q = vec![1u8; 3 * 2];
    // bad epsilon
    assert!(validate_q_matrix(&y, &obs, &q, n, 3, 2, 0.0, &CdmConfig::default()).is_err());
    assert!(validate_q_matrix(&y, &obs, &q, n, 3, 2, 1.5, &CdmConfig::default()).is_err());
    // n_attributes out of range
    assert!(validate_q_matrix(&y, &obs, &q, n, 3, 0, 0.95, &CdmConfig::default()).is_err());
    assert!(validate_q_matrix(
        &y,
        &obs,
        &[1u8; 3 * 11],
        n,
        3,
        11,
        0.95,
        &CdmConfig::default()
    )
    .is_err());
    // wrong provisional_q length
    assert!(validate_q_matrix(&y, &obs, &[1u8; 5], n, 3, 2, 0.95, &CdmConfig::default()).is_err());
    // non-binary provisional entry
    assert!(validate_q_matrix(
        &y,
        &obs,
        &[2, 0, 1, 1, 0, 1],
        n,
        3,
        2,
        0.95,
        &CdmConfig::default()
    )
    .is_err());
    assert!(validate_q_matrix(
        &y,
        &obs,
        &[0, 0, 1, 1, 0, 1],
        n,
        3,
        2,
        0.95,
        &CdmConfig::default()
    )
    .is_err());
    assert!(validate_q_matrix(
        &y[..y.len() - 1],
        &obs,
        &q,
        n,
        3,
        2,
        0.95,
        &CdmConfig::default()
    )
    .is_err());
}

#[test]
fn qval_constant_items_and_missing_cells_take_the_defined_fallback() {
    let n = 12usize;
    let y = vec![0.0; n * 3];
    let mut observed = vec![true; y.len()];
    observed[0] = false;
    let q = [1, 0, 0, 1, 1, 1];
    let cfg = CdmConfig {
        max_iter: 3,
        tol: 1e9,
        count_floor: 1e9,
        ..CdmConfig::default()
    };
    let result = validate_q_matrix(&y, &observed, &q, n, 3, 2, 0.95, &cfg).unwrap();
    assert_eq!(result.suggested_q, q);
    assert_eq!(result.suggested_pvaf, vec![0.0; 3]);
    assert_eq!(result.flagged, vec![false; 3]);

    let wald = gdina_wald_selection(&y, &observed, &q, n, 3, 2, 0.05, &cfg).unwrap();
    assert_eq!(wald.models, ["dina", "dino", "acdm", "llm", "rrum"]);
    assert_eq!(wald.selected.len(), 3);
}

#[test]
fn wald_selection_helper_covers_undefined_ties_and_parsimony() {
    assert_eq!(select_wald_model(&[0, 0], &[0.9, 0.9], 0.05, 3), -1);
    assert_eq!(select_wald_model(&[1, 1], &[f64::NAN, 0.01], 0.05, 3), -1);
    assert_eq!(select_wald_model(&[1, 1, 1], &[0.2, 0.4, 0.9], 0.05, 3), 1);
    assert_eq!(select_wald_model(&[1, 1, 1], &[0.2, 0.1, 0.9], 0.05, 1), 2);
}

#[test]
fn qval_rejects_nonconverged_calibration() {
    let n = 8usize;
    let y = vec![
        0.0, 0.0, 0.0, // 00
        0.0, 1.0, 0.0, // 01
        1.0, 0.0, 0.0, // 10
        1.0, 1.0, 1.0, // 11
        0.0, 0.0, 0.0, // repeated response patterns keep every item observed
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
    ];
    let observed = vec![true; y.len()];
    let q = vec![1, 0, 0, 1, 1, 1];
    let cfg = CdmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..CdmConfig::default()
    };

    let err = validate_q_matrix(&y, &observed, &q, n, 3, 2, 0.95, &cfg).unwrap_err();
    assert!(err.contains("did not converge"), "unexpected error: {err}");
    assert!(err.contains("1 of 1 M-steps"), "unexpected error: {err}");
    assert!(
        err.contains("tol = 1.000000e-12"),
        "unexpected error: {err}"
    );
}

/// Literature-grade Monte-Carlo (>=500 reps): recovery of the true Q-matrix by
/// PVAF validation starting from a mis-specified provisional Q, under a uniform
/// (independent) and a correlated/skew (higher-order) attribute distribution.
/// Reported as a *procedure* recovery: per-item exact q-vector rate plus
/// attribute-level true-positive / false-positive rates.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_qval_recovery_500() {
    let (k, n_items, n, reps) = (3usize, 15usize, 1000usize, 500usize);
    let truth = canonical_q3();
    let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
    let bk = [-0.6f64, 0.0, 0.6];
    let lambda = 1.5f64;
    // This simulation has N*J = 15,000 observed cells, so an absolute
    // log-likelihood increment of 2e-4 is at most 1.4e-8 per cell. Keep the
    // production iteration cap, but make the literature-grade stopping
    // contract explicit instead of accepting unfinished default-tolerance fits.
    let cfg = CdmConfig {
        tol: 2e-4,
        ..CdmConfig::default()
    };

    for &skew in [false, true].iter() {
        let (mut sum_qrec, mut sum_tpr, mut sum_fpr) = (0.0f64, 0.0f64, 0.0f64);
        let (mut max_n_iter, mut max_final_delta) = (0usize, 0.0f64);
        for rep in 0..reps {
            let seed = 0x2545F4914F6CDD1Du64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15);
            let mut rng = Lcg(seed);
            // attribute profiles
            let profiles: Vec<usize> = (0..n)
                .map(|_| {
                    if skew {
                        // correlated higher-order logistic (de la Torre & Douglas, 2004)
                        let theta = -(rng.next_f64().max(1e-12)).ln() - 1.0;
                        let mut c = 0usize;
                        for a in 0..k {
                            let pk = 1.0 / (1.0 + (-lambda * (theta - bk[a])).exp());
                            if rng.next_f64() < pk {
                                c |= 1 << a;
                            }
                        }
                        c
                    } else {
                        rng.profile(1 << k) // independent uniform over classes
                    }
                })
                .collect();
            let y = simulate(
                CdmModel::Dina,
                &truth,
                &s,
                &g,
                &profiles,
                n_items,
                k,
                &mut rng,
            );
            let observed = vec![true; n * n_items];

            // mis-specify ~1/6 of items (flip one attribute bit); the rest keep
            // the attributes identified, as the method requires.
            let mut prov = truth.clone();
            for i in 0..n_items {
                if rng.next_f64() < 0.17 {
                    let a = (rng.next_f64() * k as f64) as usize % k;
                    prov[i * k + a] ^= 1;
                }
                // guard against an all-zero provisional row (validation needs >=1)
                if (0..k).all(|a| prov[i * k + a] == 0) {
                    prov[i * k] = 1;
                }
            }
            let res = validate_q_matrix(&y, &observed, &prov, n, n_items, k, 0.95, &cfg)
                .unwrap_or_else(|err| {
                    panic!("Q validation failed for skew={skew} rep={rep} seed={seed}: {err}")
                });
            assert_eq!(res.calibration_termination_reason, "tolerance_met");
            assert_eq!(res.calibration_max_iter, cfg.max_iter);
            assert_eq!(res.calibration_tol, cfg.tol);
            assert!(
                0 < res.calibration_n_iter && res.calibration_n_iter < cfg.max_iter,
                "invalid iteration evidence for skew={skew} rep={rep} seed={seed}: {}/{}",
                res.calibration_n_iter,
                cfg.max_iter
            );
            assert!(
                res.calibration_final_loglik_change.is_finite()
                    && res.calibration_final_loglik_change < cfg.tol,
                "invalid stopping evidence for skew={skew} rep={rep} seed={seed}: {} >= {}",
                res.calibration_final_loglik_change,
                cfg.tol
            );
            max_n_iter = max_n_iter.max(res.calibration_n_iter);
            max_final_delta = max_final_delta.max(res.calibration_final_loglik_change);

            let mut qrec = 0usize;
            let (mut tp, mut fp, mut pos, mut neg) = (0usize, 0usize, 0usize, 0usize);
            for i in 0..n_items {
                if q_rows_equal(&res.suggested_q, &truth, i, k) {
                    qrec += 1;
                }
                for a in 0..k {
                    let t = truth[i * k + a] != 0;
                    let hcap = res.suggested_q[i * k + a] != 0;
                    if t {
                        pos += 1;
                        if hcap {
                            tp += 1;
                        }
                    } else {
                        neg += 1;
                        if hcap {
                            fp += 1;
                        }
                    }
                }
            }
            sum_qrec += qrec as f64 / n_items as f64;
            sum_tpr += tp as f64 / pos as f64;
            sum_fpr += fp as f64 / neg as f64;
        }
        let r = reps as f64;
        println!(
            concat!(
                "[qval MC skew={}] reps={} converged={}/{} ",
                "termination=tolerance_met iterations_max={}/{} ",
                "final_delta_max={:.6e} tol={:.1e} ",
                "q-recovery={:.3} attr-TPR={:.3} attr-FPR={:.3}"
            ),
            skew,
            reps,
            reps,
            reps,
            max_n_iter,
            cfg.max_iter,
            max_final_delta,
            cfg.tol,
            sum_qrec / r,
            sum_tpr / r,
            sum_fpr / r
        );
        assert!(
            sum_qrec / r > 0.80,
            "q-vector recovery {} skew={skew}",
            sum_qrec / r
        );
        assert!(
            sum_tpr / r > 0.90,
            "attribute TPR {} skew={skew}",
            sum_tpr / r
        );
        assert!(
            sum_fpr / r < 0.10,
            "attribute FPR {} skew={skew}",
            sum_fpr / r
        );
    }
}

// ----- CDM item-level Wald model selection (de la Torre, 2011) tests -----

/// K=2 Q with `n_single` single-attribute items per attribute (strong attribute
/// identification keeps the complete-data Wald covariance accurate) plus
/// `n_pair` two-attribute items (the ones the Wald test evaluates). The first
/// `2*n_single` items are singletons; the pair items follow.
fn wald_q2(n_single: usize, n_pair: usize) -> (Vec<u8>, usize) {
    let k = 2usize;
    let mut rows: Vec<[u8; 2]> = Vec::new();
    for _ in 0..n_single {
        rows.push([1, 0]);
    }
    for _ in 0..n_single {
        rows.push([0, 1]);
    }
    for _ in 0..n_pair {
        rows.push([1, 1]);
    }
    let n_items = rows.len();
    let mut q = vec![0u8; n_items * k];
    for (i, r) in rows.iter().enumerate() {
        q[i * k] = r[0];
        q[i * k + 1] = r[1];
    }
    (q, n_items)
}

/// CSR truth table for the K=2 scenario. Single items are 2PL-like (low/high);
/// pair items follow `kind`: DINA (conjunctive), DINO (disjunctive), A-CDM
/// (additive), or "sat" (main effects AND interaction, so no reduced model fits).
fn wald_truth(q: &[u8], n_items: usize, kind: &str) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let (item_off, qmask, kreq) = gdina_layout(q, n_items, 2);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let a = item_off[i];
        if kreq[i] == 1 {
            truth[a] = 0.15;
            truth[a + 1] = 0.85;
        } else {
            // reduce_class layout: [none, a0, a1, both]
            let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
            let (p00, p10, p01, p11) = match kind {
                "dina" => (0.15, 0.15, 0.15, 0.85), // conjunctive
                "dino" => (0.15, 0.85, 0.85, 0.85), // disjunctive (any mastered -> 1-s)
                "acdm" => (0.10, 0.45, 0.45, 0.80), // additive 0.1 + .35a0 + .35a1
                // LLM: additive on the logit, logit(P) = -3 + 2 a0 + 2 a1. Chosen
                // asymmetric (2*(-3)+2+2 = -2 != 0) so the four points are NOT
                // reflection-symmetric about 0 -> genuinely identity-NONadditive
                // (A-CDM must reject) yet exactly logit-additive (LLM must not). Also
                // log-nonadditive (P10/P00 != P11/P01), so R-RUM rejects too.
                "llm" => (sig(-3.0), sig(-1.0), sig(-1.0), sig(1.0)),
                // R-RUM: additive on the log, P = pi* r0^(1-a0) r1^(1-a1) with
                // pi*=0.92, r0=0.3, r1=0.4. Log-additive (P10/P00 = P11/P01 = 1/r0)
                // but strongly identity- AND logit-NONadditive (the high pi* makes
                // logit(P) depart from log(P) sharply), so only R-RUM survives.
                "rrum" => (0.92 * 0.3 * 0.4, 0.92 * 0.4, 0.92 * 0.3, 0.92),
                _ => (0.10, 0.35, 0.35, 0.90), // main effects + interaction (saturated)
            };
            truth[a] = p00;
            truth[a + 1] = p10;
            truth[a + 2] = p01;
            truth[a + 3] = p11;
        }
    }
    (item_off, qmask, truth)
}

/// DINA-generated pair items are classified as DINA (the conjunctive reduced
/// model is not rejected while the additive one is).
#[test]
fn wald_dina_data_selects_dina() {
    let (q, n_items) = wald_q2(5, 8);
    let n = 5000usize;
    let first_pair = 10usize;
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "dina");
    let mut rng = Lcg(4011);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    assert_eq!(
        res.models,
        vec![
            "dina".to_string(),
            "dino".to_string(),
            "acdm".to_string(),
            "llm".to_string(),
            "rrum".to_string(),
        ]
    );
    let nm = res.models.len();
    let pair_dina = (first_pair..n_items)
        .filter(|&i| res.selected[i] == 0)
        .count();
    assert!(pair_dina >= 7, "DINA selected for {pair_dina}/8 pair items");
    // single-attribute items are trivial (df=0) -> saturated, NaN stats
    for i in 0..first_pair {
        assert_eq!(res.selected[i], -1);
        assert!(res.wald_stat[i * nm].is_nan());
    }
}

/// DINO-generated pair items are classified as DINO (the disjunctive reduced
/// model is not rejected while DINA and A-CDM are). Exercises the general
/// (non-coordinate) linear restriction and the DINA/DINO parameter-count tie.
#[test]
fn wald_dino_data_selects_dino() {
    let (q, n_items) = wald_q2(5, 8);
    let n = 8000usize;
    let first_pair = 10usize;
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "dino");
    let mut rng = Lcg(6060);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let nm = res.models.len();
    let pair_dino = (first_pair..n_items)
        .filter(|&i| res.selected[i] == 1)
        .count();
    assert!(pair_dino >= 7, "DINO selected for {pair_dino}/8 pair items");
    // DINO and DINA both have df = 2^K - 2 = 2 at K=2
    assert_eq!(res.wald_df[first_pair * nm], 2); // DINA
    assert_eq!(res.wald_df[first_pair * nm + 1], 2); // DINO
}

/// Additive-generated pair items are classified as A-CDM (additive not rejected,
/// conjunctive DINA and disjunctive DINO rejected). A-CDM is candidate index 2.
#[test]
fn wald_acdm_data_selects_acdm() {
    let (q, n_items) = wald_q2(5, 8);
    let n = 5000usize;
    let first_pair = 10usize;
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "acdm");
    let mut rng = Lcg(2027);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let pair_acdm = (first_pair..n_items)
        .filter(|&i| res.selected[i] == 2)
        .count();
    assert!(
        pair_acdm >= 7,
        "A-CDM selected for {pair_acdm}/8 pair items"
    );
}

/// Faithfulness anchor for the link-transformed reduced models. The LLM and R-RUM
/// truths are constructed to be additive ONLY on their own link (logit / log) and
/// genuinely NON-additive on the identity link, so a correct implementation must
/// (a) select LLM (index 3) / R-RUM (index 4) and (b) *reject* the identity-link
/// A-CDM (index 2) — a sign/identity bug in the Jacobian covariance or the
/// transformed delta would collapse this distinction. This is deliberately a
/// non-centered, non-trivial truth: A-CDM, LLM and R-RUM all cost 1+K parameters,
/// so only the transform can break the tie.
#[test]
fn wald_llm_and_rrum_data_select_their_link() {
    let (q, n_items) = wald_q2(5, 8);
    let n = 8000usize;
    let first_pair = 10usize;

    // LLM truth (logit-additive; identity- and log-NONadditive) -> LLM selected.
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "llm");
    let mut rng = Lcg(770011);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let nm = res.models.len();
    let pair_llm = (first_pair..n_items)
        .filter(|&i| res.selected[i] == 3)
        .count();
    assert!(pair_llm >= 7, "LLM selected for {pair_llm}/8 pair items");
    // The identity-link A-CDM must be rejected on these identity-nonadditive items.
    let acdm_rej = (first_pair..n_items)
        .filter(|&i| res.p_value[i * nm + 2] < 0.05)
        .count();
    assert!(
        acdm_rej >= 7,
        "A-CDM rejected on {acdm_rej}/8 LLM items (identity-nonadditive)"
    );

    // R-RUM truth (log-additive; identity- and logit-NONadditive) -> R-RUM selected.
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "rrum");
    let mut rng = Lcg(880022);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let pair_rrum = (first_pair..n_items)
        .filter(|&i| res.selected[i] == 4)
        .count();
    assert!(
        pair_rrum >= 7,
        "R-RUM selected for {pair_rrum}/8 pair items"
    );
    // The logit-link LLM must be rejected on these logit-nonadditive items.
    let llm_rej = (first_pair..n_items)
        .filter(|&i| res.p_value[i * nm + 3] < 0.05)
        .count();
    assert!(
        llm_rej >= 7,
        "LLM rejected on {llm_rej}/8 R-RUM items (logit-nonadditive)"
    );
}

/// Items with both main effects and an interaction reject every reduced model,
/// so the saturated G-DINA is kept.
#[test]
fn wald_saturated_data_selects_saturated() {
    let (q, n_items) = wald_q2(5, 8);
    let n = 5000usize;
    let first_pair = 10usize;
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "sat");
    let mut rng = Lcg(9091);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let nm = res.models.len();
    let pair_sat = (first_pair..n_items)
        .filter(|&i| res.selected[i] == -1)
        .count();
    assert!(pair_sat >= 7, "saturated kept for {pair_sat}/8 pair items");
    // every reduced model (DINA/DINO/A-CDM/LLM/R-RUM) carries a positive, finite stat
    for i in first_pair..n_items {
        for m in 0..nm {
            assert!(res.wald_stat[i * nm + m].is_finite() && res.wald_stat[i * nm + m] >= 0.0);
            assert!(res.p_value[i * nm + m].is_finite());
        }
    }
}

/// Degrees of freedom are exactly the restriction sizes: DINA & DINO df = 2^K-2,
/// A-CDM df = 2^K-1-K, for K=3 items.
#[test]
fn wald_degrees_of_freedom() {
    // K=3 Q: single items (identification) + one triple item to read df off.
    let k = 3usize;
    let mut rows: Vec<[u8; 3]> = Vec::new();
    for a in 0..3 {
        for _ in 0..3 {
            let mut r = [0u8; 3];
            r[a] = 1;
            rows.push(r);
        }
    }
    rows.push([1, 1, 1]); // one K=3 item
    let n_items = rows.len();
    let mut q = vec![0u8; n_items * k];
    for (i, r) in rows.iter().enumerate() {
        q[i * k..i * k + k].copy_from_slice(r);
    }
    let n = 3000usize;
    let (item_off, qmask, _kr) = gdina_layout(&q, n_items, k);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let a = item_off[i];
        let w = item_off[i + 1] - a;
        for l in 0..w {
            truth[a + l] = 0.15 + 0.7 * (l.count_ones() as f64) / (w.trailing_zeros() as f64);
        }
    }
    let mut rng = Lcg(31337);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let res = gdina_wald_selection(
        &y,
        &observed,
        &q,
        n,
        n_items,
        k,
        0.05,
        &CdmConfig::default(),
    )
    .unwrap();
    let nm = res.models.len();
    let triple = n_items - 1;
    assert_eq!(res.wald_df[triple * nm], (1 << k) - 2, "DINA df"); // 6
    assert_eq!(res.wald_df[triple * nm + 1], (1 << k) - 2, "DINO df"); // 6
    assert_eq!(res.wald_df[triple * nm + 2], (1 << k) - 1 - k, "A-CDM df"); // 4
    assert_eq!(res.wald_df[triple * nm + 3], (1 << k) - 1 - k, "LLM df"); // 4
    assert_eq!(res.wald_df[triple * nm + 4], (1 << k) - 1 - k, "R-RUM df"); // 4
                                                                            // single-attribute items: no test (df=0), saturated
    assert_eq!(res.wald_df[0], 0);
    assert_eq!(res.selected[0], -1);
}

#[test]
fn wald_rejects_malformed() {
    let (q, n_items) = wald_q2(2, 2);
    let n = 10usize;
    let y = vec![0.0f64; n * n_items];
    let obs = vec![true; n * n_items];
    // alpha out of (0,1)
    assert!(gdina_wald_selection(&y, &obs, &q, n, n_items, 2, 0.0, &CdmConfig::default()).is_err());
    assert!(gdina_wald_selection(&y, &obs, &q, n, n_items, 2, 1.0, &CdmConfig::default()).is_err());
    // shape errors are delegated to fit_gdina's validate
    assert!(gdina_wald_selection(
        &y[..5],
        &obs,
        &q,
        n,
        n_items,
        2,
        0.05,
        &CdmConfig::default()
    )
    .is_err());
}

#[test]
fn wald_rejects_nonconverged_gdina_calibration() {
    let (q, n_items) = wald_q2(2, 2);
    let n = 80usize;
    let mut rng = Lcg(20260715);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "dina");
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = CdmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..CdmConfig::default()
    };

    let err = gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &cfg)
        .expect_err("Wald selection must not use unfinished G-DINA parameters");
    assert!(err.contains("G-DINA calibration did not converge after 1 of 1 M-steps"));
    assert!(err.contains("final |delta loglik| ="));
    assert!(err.contains("tol = 1.000000e-12"));
}

/// Literature-grade Monte-Carlo (>=500 reps): Type I error (reject the TRUE
/// reduced model ~ alpha) and power (reject a false, over-restrictive model),
/// under uniform and correlated/skew attribute distributions.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_wald_type1_power_500() {
    let reps = 500usize;
    let (q, n_items) = wald_q2(5, 8);
    let n = 3000usize;
    let first_pair = 10usize;
    let k = 2usize;
    let bk = [-0.4f64, 0.4];
    let lambda = 1.5f64;
    let draw_profiles = |rng: &mut Lcg, skew: bool| -> Vec<usize> {
        (0..n)
            .map(|_| {
                if skew {
                    let theta = -(rng.next_f64().max(1e-12)).ln() - 1.0;
                    let mut c = 0usize;
                    for a in 0..k {
                        let pk = 1.0 / (1.0 + (-lambda * (theta - bk[a])).exp());
                        if rng.next_f64() < pk {
                            c |= 1 << a;
                        }
                    }
                    c
                } else {
                    rng.profile(1 << k)
                }
            })
            .collect()
    };

    // Candidate columns: DINA=0, DINO=1, A-CDM=2, LLM=3, R-RUM=4.
    for &skew in [false, true].iter() {
        let (mut t1_acdm, mut t1_dina, mut t1_dino, mut t1_llm, mut t1_rrum) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        // Power of over-restrictive models against each additive-family truth: the
        // identity-link A-CDM and cross-link LLM/R-RUM must reject the wrong link.
        let (mut pow_dina, mut pow_dino, mut pow_acdm_llm, mut pow_rrum_llm, mut pow_llm_rrum) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut den = 0.0f64;
        for rep in 0..reps {
            let mut rng = Lcg(0x9E3779B97F4A7C15u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03));
            let obs = vec![true; n * n_items];
            let run = |kind: &str, rng: &mut Lcg| {
                let (io, qm, tr) = wald_truth(&q, n_items, kind);
                let prof = draw_profiles(rng, skew);
                let y = simulate_gdina(&qm, &io, &tr, &prof, n_items, rng);
                gdina_wald_selection(&y, &obs, &q, n, n_items, k, 0.05, &CdmConfig::default())
                    .unwrap()
            };
            // A-CDM truth: Type I of A-CDM (col 2) + power of the false DINA (col 0).
            let ra = run("acdm", &mut rng);
            // DINA truth: Type I of DINA (col 0) + power of the false DINO (col 1).
            let rd = run("dina", &mut rng);
            // DINO truth: Type I of DINO (col 1).
            let rn = run("dino", &mut rng);
            // LLM truth: Type I of LLM (col 3) + power of the false identity A-CDM
            // (col 2) and false log-link R-RUM (col 4).
            let rl = run("llm", &mut rng);
            // R-RUM truth: Type I of R-RUM (col 4) + power of the false logit LLM (col 3).
            let rr = run("rrum", &mut rng);
            let nm = ra.models.len();
            for i in first_pair..n_items {
                if ra.p_value[i * nm + 2] < 0.05 {
                    t1_acdm += 1.0;
                }
                if ra.p_value[i * nm] < 0.05 {
                    pow_dina += 1.0; // DINA false under A-CDM truth
                }
                if rd.p_value[i * nm] < 0.05 {
                    t1_dina += 1.0;
                }
                if rd.p_value[i * nm + 1] < 0.05 {
                    pow_dino += 1.0; // DINO false under DINA truth
                }
                if rn.p_value[i * nm + 1] < 0.05 {
                    t1_dino += 1.0;
                }
                if rl.p_value[i * nm + 3] < 0.05 {
                    t1_llm += 1.0;
                }
                if rl.p_value[i * nm + 2] < 0.05 {
                    pow_acdm_llm += 1.0; // A-CDM false under LLM truth
                }
                if rl.p_value[i * nm + 4] < 0.05 {
                    pow_rrum_llm += 1.0; // R-RUM false under LLM truth
                }
                if rr.p_value[i * nm + 4] < 0.05 {
                    t1_rrum += 1.0;
                }
                if rr.p_value[i * nm + 3] < 0.05 {
                    pow_llm_rrum += 1.0; // LLM false under R-RUM truth
                }
                den += 1.0;
            }
        }
        println!(
            "[wald MC skew={skew}] reps={reps} TypeI(dina)={:.3} TypeI(dino)={:.3} \
             TypeI(acdm)={:.3} TypeI(llm)={:.3} TypeI(rrum)={:.3} power(dina|acdm)={:.3} \
             power(dino|dina)={:.3} power(acdm|llm)={:.3} power(rrum|llm)={:.3} \
             power(llm|rrum)={:.3}",
            t1_dina / den,
            t1_dino / den,
            t1_acdm / den,
            t1_llm / den,
            t1_rrum / den,
            pow_dina / den,
            pow_dino / den,
            pow_acdm_llm / den,
            pow_rrum_llm / den,
            pow_llm_rrum / den
        );
        // Complete-data covariance is mildly liberal; allow up to ~2.5x nominal.
        assert!(t1_acdm / den < 0.13, "A-CDM Type I {}", t1_acdm / den);
        assert!(t1_dina / den < 0.13, "DINA Type I {}", t1_dina / den);
        assert!(t1_dino / den < 0.13, "DINO Type I {}", t1_dino / den);
        assert!(t1_llm / den < 0.13, "LLM Type I {}", t1_llm / den);
        assert!(t1_rrum / den < 0.13, "R-RUM Type I {}", t1_rrum / den);
        assert!(pow_dina / den > 0.95, "DINA power {}", pow_dina / den);
        assert!(pow_dino / den > 0.95, "DINO power {}", pow_dino / den);
        assert!(
            pow_acdm_llm / den > 0.95,
            "A-CDM|LLM power {}",
            pow_acdm_llm / den
        );
        assert!(
            pow_rrum_llm / den > 0.90,
            "R-RUM|LLM power {}",
            pow_rrum_llm / den
        );
        assert!(
            pow_llm_rrum / den > 0.90,
            "LLM|R-RUM power {}",
            pow_llm_rrum / den
        );
    }
}

// ----- Higher-order structured attribute prior (de la Torre & Douglas, 2004) -----

/// Simulate higher-order DINA data: theta -> attribute mastery via
/// sigmoid(a_k theta + d_k), then the DINA gate with slip/guess.
#[allow(clippy::too_many_arguments)]
fn simulate_ho_dina(
    a: &[f64],
    d: &[f64],
    s: &[f64],
    g: &[f64],
    q: &[u8],
    n: usize,
    n_items: usize,
    n_attr: usize,
    skew: bool,
    rng: &mut Lcg,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut y = vec![0.0f64; n * n_items];
    let mut profiles = vec![0usize; n];
    let mut thetas = vec![0.0f64; n];
    for j in 0..n {
        let theta = if skew {
            // standardized shifted chi-square(3): mean 0, var 1, right-skewed
            let mut cc = 0.0;
            for _ in 0..3 {
                let z = rng.normal();
                cc += z * z;
            }
            (cc - 3.0) / (6.0_f64).sqrt()
        } else {
            rng.normal()
        };
        thetas[j] = theta;
        let mut c = 0usize;
        for k in 0..n_attr {
            let p = 1.0 / (1.0 + (-(a[k] * theta + d[k])).exp());
            if rng.next_f64() < p {
                c |= 1 << k;
            }
        }
        profiles[j] = c;
        for i in 0..n_items {
            let mask = qmask_of(q, i, n_attr);
            let eta = (c & mask) == mask;
            let p = if eta { 1.0 - s[i] } else { g[i] };
            y[j * n_items + i] = rng.bern(p);
        }
    }
    (y, profiles, thetas)
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// ANCHOR: with every attribute slope zero, the implied class prior is exactly the
/// independent-attribute Bernoulli product (theta drops out), bit-for-bit.
#[test]
fn ho_pi_independent_when_slope_zero() {
    let k = 3usize;
    let a = vec![0.0f64; k];
    let d = vec![0.7f64, -0.4, 0.2];
    let pi = ho_pi_from_params(&a, &d, k);
    let pk: Vec<f64> = d.iter().map(|&dk| 1.0 / (1.0 + (-dk).exp())).collect();
    for c in 0..(1 << k) {
        let mut prod = 1.0f64;
        for (bit, &p) in pk.iter().enumerate() {
            prod *= if (c >> bit) & 1 == 1 { p } else { 1.0 - p };
        }
        assert!(
            (pi[c] - prod).abs() < 1e-12,
            "class {c}: {} vs {}",
            pi[c],
            prod
        );
    }
    assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

/// Higher-order DINA recovery: attribute slopes/intercepts, slip/guess, the trait,
/// and attribute classification under a known higher-order structure.
#[test]
fn ho_recovers_params() {
    let (n_attr, n_items, n) = (3usize, 15usize, 4000usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..n_items {
        // 4 single-attribute items per attribute + 3 pair items
        if i < 12 {
            q[i * n_attr + (i / 4)] = 1;
        } else {
            q[i * n_attr + (i - 12)] = 1;
            q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
        }
    }
    let a_true = vec![1.2f64, 1.5, 0.9];
    let d_true = vec![0.3f64, -0.5, 0.6];
    let s = vec![0.12f64; n_items];
    let g = vec![0.12f64; n_items];
    let mut rng = Lcg(70424);
    let (y, profiles, thetas) = simulate_ho_dina(
        &a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_ho_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace));
    assert!(res.n_parameters == 2 * n_items + 2 * n_attr);
    assert!((res.profile_prob.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // slip/guess
    assert!(
        rmse(&res.slip, &s) < 0.05,
        "slip RMSE {}",
        rmse(&res.slip, &s)
    );
    assert!(
        rmse(&res.guess, &g) < 0.05,
        "guess RMSE {}",
        rmse(&res.guess, &g)
    );
    // higher-order parameters (identified up to the N(0,1) trait scale)
    assert!(
        rmse(&res.attr_slope, &a_true) < 0.4,
        "a RMSE {}",
        rmse(&res.attr_slope, &a_true)
    );
    assert!(
        rmse(&res.attr_intercept, &d_true) < 0.3,
        "d RMSE {}",
        rmse(&res.attr_intercept, &d_true)
    );
    assert!(res.attr_slope.iter().all(|&x| x > 0.0));
    // trait recovery (EAP is shrunk, so correlation is the right metric)
    assert!(
        corr(&res.theta, &thetas) > 0.6,
        "theta corr {}",
        corr(&res.theta, &thetas)
    );
    // attribute classification
    assert!(
        attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85,
        "attribute agreement {}",
        attribute_agreement(&res.attr_prob, &profiles, n, n_attr)
    );
}

/// Data from independent attributes (all true slopes 0) -> the *implied class
/// distribution* `pi_c` recovers the independent-attribute product. (The
/// individual slopes are not the right target: independence is also consistent
/// with a single nonzero slope, since one attribute loading on theta induces no
/// cross-attribute correlation. The likelihood identifies only `pi_c`.)
#[test]
fn ho_independent_data_recovers_pi() {
    let (n_attr, n_items, n) = (3usize, 15usize, 4000usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..n_items {
        if i < 12 {
            q[i * n_attr + (i / 4)] = 1;
        } else {
            q[i * n_attr + (i - 12)] = 1;
            q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
        }
    }
    let a_true = vec![0.0f64; n_attr];
    let d_true = vec![0.4f64, -0.3, 0.2];
    let s = vec![0.1f64; n_items];
    let g = vec![0.1f64; n_items];
    let mut rng = Lcg(9021);
    let (y, _p, _t) = simulate_ho_dina(
        &a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_ho_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    let pi_true = ho_pi_from_params(&a_true, &d_true, n_attr);
    assert!(
        rmse(&res.profile_prob, &pi_true) < 0.03,
        "implied pi RMSE {}",
        rmse(&res.profile_prob, &pi_true)
    );
}

/// Single-attribute Q: DINA and DINO share the ideal-response gate, so the
/// higher-order fits coincide. Also exercises missing-at-random data.
#[test]
fn ho_reduces_dino_and_handles_missing() {
    let (n_attr, n_items, n) = (2usize, 8usize, 1000usize);
    let q: Vec<u8> = (0..n_items)
        .flat_map(|i| if i % 2 == 0 { [1u8, 0] } else { [0u8, 1] })
        .collect();
    let a_true = vec![1.0f64, 1.0];
    let d_true = vec![0.0f64, 0.0];
    let s = vec![0.15f64; n_items];
    let g = vec![0.15f64; n_items];
    let mut rng = Lcg(4242);
    let (mut y, _p, _t) = simulate_ho_dina(
        &a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng,
    );
    let mut observed = vec![true; n * n_items];
    // DINA == DINO on single-attribute items
    let da = fit_ho_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    let di = fit_ho_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dino,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(rmse(&da.slip, &di.slip) < 1e-9 && rmse(&da.guess, &di.guess) < 1e-9);
    // missing-at-random cells dropped, still converges
    for o in observed.iter_mut() {
        if rng.next_f64() < 0.15 {
            *o = false;
        }
    }
    for (idx, o) in observed.iter().enumerate() {
        if !o {
            y[idx] = 0.0;
        }
    }
    let rm = fit_ho_cdm(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        CdmModel::Dina,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(rm.loglik_trace.iter().all(|v| v.is_finite()));
}

/// Full structural Newton steps used to make the observed log-likelihood fall
/// (seed 12) and could then satisfy `abs(delta) < tol` on a negative change,
/// falsely reporting convergence (seed 6).
#[test]
fn ho_structural_newton_preserves_em_ascent() {
    let (n_attr, n_items, n) = (3usize, 9usize, 40usize);
    let q = vec![
        1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
    ];
    let item_prob = [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    for (seed, max_iter) in [(12u64, 100usize), (6, 500)] {
        let mut rng = Lcg(seed);
        let mut y = vec![0.0; n * n_items];
        for j in 0..n {
            for i in 0..n_items {
                y[j * n_items + i] = rng.bern(item_prob[i]);
            }
        }
        let observed = vec![true; y.len()];
        let cfg = CdmConfig {
            max_iter,
            ..CdmConfig::default()
        };
        let res = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg).unwrap();
        let final_delta = res.loglik_trace[res.loglik_trace.len() - 1]
            - res.loglik_trace[res.loglik_trace.len() - 2];
        assert_eq!(res.max_iter, cfg.max_iter);
        assert_eq!(res.stopping_tolerance, cfg.tol);
        assert_eq!(res.final_loglik_change, final_delta);
        assert!(
            nondecreasing(&res.loglik_trace),
            "higher-order GEM lowered log-likelihood for seed {seed}: {:?}",
            res.loglik_trace
        );
        if seed == 6 {
            assert!(res.converged, "safeguarded seed-6 fit did not converge");
            assert_eq!(res.termination_reason, "tolerance_met");
            assert!(res.n_iter < res.max_iter);
            assert!(
                (0.0..cfg.tol).contains(&res.final_loglik_change),
                "convergence must be a non-negative improvement below tol; delta={:e}",
                res.final_loglik_change
            );
        } else {
            assert!(!res.converged);
            assert_eq!(res.termination_reason, "max_iter_reached");
            assert_eq!(res.n_iter, res.max_iter);
            assert!(res.final_loglik_change.is_finite());
            assert!(res.final_loglik_change >= res.stopping_tolerance);
        }
    }
}

#[test]
fn ho_validate_rejects_malformed() {
    let cfg = CdmConfig::default();
    // y length mismatch (expects n_persons * n_items = 2)
    assert!(fit_ho_cdm(&[0.0], &[true], &[1, 1], 1, 2, 1, CdmModel::Dina, &cfg).is_err());
    // all-zero Q column: attribute 1 measured by no item
    assert!(fit_ho_cdm(
        &[0.0, 1.0],
        &[true, true],
        &[1, 0, 1, 0],
        1,
        2,
        2,
        CdmModel::Dina,
        &cfg
    )
    .is_err());
}

/// Literature-grade Monte-Carlo (>=500 reps): higher-order DINA parameter recovery
/// under normal and skew (mis-specified prior) trait distributions.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_ho_recovery_500() {
    let (n_attr, n_items, n, reps) = (3usize, 15usize, 1000usize, 500usize);
    let mut q = vec![0u8; n_items * n_attr];
    for i in 0..n_items {
        if i < 12 {
            q[i * n_attr + (i / 4)] = 1;
        } else {
            q[i * n_attr + (i - 12)] = 1;
            q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
        }
    }
    let a_true = vec![1.2f64, 1.5, 0.9];
    let d_true = vec![0.3f64, -0.5, 0.6];
    let s = vec![0.12f64; n_items];
    let g = vec![0.12f64; n_items];
    for &skew in [false, true].iter() {
        let (mut ra, mut rd, mut ba, mut bd, mut attr, mut nconv) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        for rep in 0..reps {
            let mut rng = Lcg(0xA24BAED4963EE407u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15));
            let (y, profiles, _t) = simulate_ho_dina(
                &a_true, &d_true, &s, &g, &q, n, n_items, n_attr, skew, &mut rng,
            );
            let observed = vec![true; n * n_items];
            let res = fit_ho_cdm(
                &y,
                &observed,
                &q,
                n,
                n_items,
                n_attr,
                CdmModel::Dina,
                &CdmConfig::default(),
            )
            .unwrap();
            if res.converged {
                nconv += 1;
                ra += rmse(&res.attr_slope, &a_true);
                rd += rmse(&res.attr_intercept, &d_true);
                ba += bias(&res.attr_slope, &a_true);
                bd += bias(&res.attr_intercept, &d_true);
                attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
            }
        }
        let conv_rate = nconv as f64 / reps as f64;
        assert!(
            conv_rate >= 0.95,
            "higher-order MC convergence rate {conv_rate:.3} below 0.95 for skew={skew}"
        );
        let den = nconv as f64;
        ra /= den;
        rd /= den;
        ba /= den;
        bd /= den;
        attr /= den;
        println!(
            "[HO-DINA MC skew={skew}] reps={reps} converged={nconv} ({conv_rate:.3}) \
             RMSE(a)={ra:.3} RMSE(d)={rd:.3} bias(a)={ba:.3} bias(d)={bd:.3} \
             attr-agree={attr:.3}"
        );
        // The trait prior is fixed N(0,1); under a skewed true trait the
        // structural slope/intercept degrade (prior mis-specification, as in 2PL
        // MMLE), while the attribute classification stays robust. Observed:
        // normal RMSE(a)~0.28 / RMSE(d)~0.09; skew RMSE(a)~0.37 / RMSE(d)~0.18;
        // attribute agreement ~0.98 in both. Bounds are condition-specific.
        let (a_bound, d_bound) = if skew { (0.45, 0.25) } else { (0.32, 0.15) };
        assert!(ra < a_bound, "RMSE(a) {ra} skew={skew}");
        assert!(rd < d_bound, "RMSE(d) {rd} skew={skew}");
        assert!(attr > 0.90, "attribute agreement {attr} skew={skew}");
    }
}

// ----- Higher-order G-DINA (de la Torre & Douglas, 2004 x de la Torre, 2011) -----

/// Simulate higher-order G-DINA data: theta -> attribute mastery via
/// sigmoid(a_k theta + d_k), then draw responses from the SATURATED per-reduced-
/// class truth table (CSR, indexed by reduce_class), returning (y, profiles, thetas).
#[allow(clippy::too_many_arguments)]
fn simulate_ho_gdina(
    a: &[f64],
    d: &[f64],
    qmask: &[usize],
    item_off: &[usize],
    truth_p: &[f64],
    n: usize,
    n_items: usize,
    n_attr: usize,
    skew: bool,
    rng: &mut Lcg,
) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
    let mut y = vec![0.0f64; n * n_items];
    let mut profiles = vec![0usize; n];
    let mut thetas = vec![0.0f64; n];
    for j in 0..n {
        let theta = if skew {
            let mut cc = 0.0;
            for _ in 0..3 {
                let z = rng.normal();
                cc += z * z;
            }
            (cc - 3.0) / (6.0_f64).sqrt()
        } else {
            rng.normal()
        };
        thetas[j] = theta;
        let mut c = 0usize;
        for k in 0..n_attr {
            let pk = 1.0 / (1.0 + (-(a[k] * theta + d[k])).exp());
            if rng.next_f64() < pk {
                c |= 1 << k;
            }
        }
        profiles[j] = c;
        for i in 0..n_items {
            let l = reduce_class(c, qmask[i]);
            y[j * n_items + i] = rng.bern(truth_p[item_off[i] + l]);
        }
    }
    (y, profiles, thetas)
}

/// A canonical K=3 Q: single-attribute items (identification) + pair + triple.
fn hogdina_q3() -> Vec<u8> {
    let k = 3usize;
    let mut q = vec![0u8; 15 * k];
    let rows: [&[usize]; 15] = [
        &[0],
        &[1],
        &[2],
        &[0],
        &[1],
        &[2],
        &[0],
        &[1],
        &[2], // 9 singles
        &[0, 1],
        &[1, 2],
        &[0, 2],
        &[0, 1],
        &[1, 2],    // 5 pairs
        &[0, 1, 2], // 1 triple
    ];
    for (i, r) in rows.iter().enumerate() {
        for &at in *r {
            q[i * k + at] = 1;
        }
    }
    q
}

/// NON-TRIVIAL anchor: HO structure with SATURATED item probs set to the DINA
/// pattern (g off-top, 1-s at top). The free saturated fit recovers those probs
/// (so the item-level identity-link delta shows the DINA pattern) and the
/// higher-order (a, d).
#[test]
fn ho_gdina_recovers_dina_pattern() {
    let (n_attr, n_items, n) = (3usize, 15usize, 3000usize);
    let q = hogdina_q3();
    let (item_off, qmask, _kreq) = gdina_layout(&q, n_items, n_attr);
    let (s, g) = (0.15f64, 0.2f64);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a0, b0) = (item_off[i], item_off[i + 1]);
        for l in a0..b0 {
            truth[l] = g;
        }
        truth[b0 - 1] = 1.0 - s; // DINA: only the all-mastered reduced class is high
    }
    let a_true = vec![1.2f64, 1.5, 0.9];
    let d_true = vec![0.3f64, -0.5, 0.6];
    let mut rng = Lcg(20242011);
    let (y, profiles, thetas) = simulate_ho_gdina(
        &a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, false, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace));
    assert!(res.n_parameters == item_off[n_items] + 2 * n_attr);
    // saturated item probs recover the DINA pattern
    assert!(
        rmse(&res.item_prob, &truth) < 0.04,
        "item p RMSE {}",
        rmse(&res.item_prob, &truth)
    );
    // identity-link delta: intercept ~ g, top interaction ~ (1-s)-g, interior ~ 0
    for i in 0..n_items {
        let (a0, b0) = (item_off[i], item_off[i + 1]);
        let dl = &res.item_delta[a0..b0];
        assert!((dl[0] - g).abs() < 0.06, "delta0 item {i}");
        assert!(
            (dl[b0 - a0 - 1] - ((1.0 - s) - g)).abs() < 0.06,
            "delta_full item {i}"
        );
        for l in 1..(b0 - a0 - 1) {
            assert!(dl[l].abs() < 0.06, "interior delta item {i} idx {l}");
        }
    }
    // higher-order recovery (identified at K=3) + trait + classification
    assert!(
        rmse(&res.attr_slope, &a_true) < 0.45,
        "a RMSE {}",
        rmse(&res.attr_slope, &a_true)
    );
    assert!(res.attr_slope.iter().all(|&x| x > 0.0));
    assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.9);
    let tc = {
        let corr = |x: &[f64], y: &[f64]| {
            let nn = x.len() as f64;
            let (mx, my) = (x.iter().sum::<f64>() / nn, y.iter().sum::<f64>() / nn);
            let (mut sxy, mut sx, mut sy) = (0.0, 0.0, 0.0);
            for i in 0..x.len() {
                sxy += (x[i] - mx) * (y[i] - my);
                sx += (x[i] - mx).powi(2);
                sy += (y[i] - my).powi(2);
            }
            sxy / (sx.sqrt() * sy.sqrt())
        };
        corr(&res.theta, &thetas)
    };
    assert!(tc > 0.55, "theta corr {tc}");
}

/// Independent-attribute data (all slopes 0) -> the implied class distribution
/// recovers the independent-attribute product (K=3; the identified quantity).
#[test]
fn ho_gdina_independent_recovers_pi() {
    let (n_attr, n_items, n) = (3usize, 15usize, 3000usize);
    let q = hogdina_q3();
    let (item_off, qmask, _kr) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a0, b0) = (item_off[i], item_off[i + 1]);
        for (li, l) in (a0..b0).enumerate() {
            truth[l] = 0.15 + 0.7 * (li.count_ones() as f64) / (b0 - a0).trailing_zeros() as f64;
        }
    }
    let a_true = vec![0.0f64; n_attr];
    let d_true = vec![0.4f64, -0.3, 0.2];
    let mut rng = Lcg(7777);
    let (y, _p, _t) = simulate_ho_gdina(
        &a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, false, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    let pi_true = ho_pi_from_params(&a_true, &d_true, n_attr);
    assert!(
        res.converged,
        "termination={} n_iter={} relative_change={} tolerance={} attr_slope={:?}",
        res.termination_reason,
        res.n_iter,
        res.final_relative_loglik_change,
        res.stopping_tolerance,
        res.attr_slope
    );
    assert_eq!(res.termination_reason, "tolerance_met");
    assert!(res.final_relative_loglik_change < res.stopping_tolerance);
    assert!(nondecreasing(&res.loglik_trace));
    println!(
        "[HO-GDINA independent] n_iter={} delta_loglik={:.3e} relative_delta={:.3e} tol={:.1e}",
        res.n_iter,
        res.final_loglik_change,
        res.final_relative_loglik_change,
        res.stopping_tolerance
    );
    assert!(
        rmse(&res.profile_prob, &pi_true) < 0.03,
        "pi RMSE {}",
        rmse(&res.profile_prob, &pi_true)
    );
}

#[test]
fn ho_gdina_handles_missing_and_validates() {
    let (n_attr, n_items, n) = (3usize, 15usize, 1000usize);
    let q = hogdina_q3();
    let (item_off, qmask, _kr) = gdina_layout(&q, n_items, n_attr);
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a0, b0) = (item_off[i], item_off[i + 1]);
        for l in a0..b0 {
            truth[l] = 0.2;
        }
        truth[b0 - 1] = 0.85;
    }
    let mut rng = Lcg(99);
    let (mut y, _p, _t) = simulate_ho_gdina(
        &[1.0, 1.0, 1.0],
        &[0.0, 0.0, 0.0],
        &qmask,
        &item_off,
        &truth,
        n,
        n_items,
        n_attr,
        false,
        &mut rng,
    );
    let mut observed = vec![true; n * n_items];
    for o in observed.iter_mut() {
        if rng.next_f64() < 0.15 {
            *o = false;
        }
    }
    for (idx, o) in observed.iter().enumerate() {
        if !o {
            y[idx] = 0.0;
        }
    }
    let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
    assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
    // malformed
    let cfg = CdmConfig::default();
    assert!(fit_ho_gdina(&[0.0], &[true], &[1, 1], 1, 2, 1, &cfg).is_err()); // y length mismatch
    assert!(fit_ho_gdina(&[0.0, 1.0], &[true, true], &[0, 0, 0, 0], 1, 2, 2, &cfg).is_err()); // all-zero Q row
    let err = fit_ho_gdina(
        &[0.0, 1.0, 1.0, 0.0],
        &[true; 4],
        &[1, 0, 0, 1],
        2,
        2,
        2,
        &cfg,
    )
    .unwrap_err();
    assert!(err.contains("at least 3 attributes"), "{err}");

    let one_step = fit_ho_gdina(
        &y,
        &observed,
        &q,
        n,
        n_items,
        n_attr,
        &CdmConfig {
            max_iter: 1,
            tol: 1e-12,
            ..CdmConfig::default()
        },
    )
    .unwrap();
    assert!(!one_step.converged);
    assert_eq!(one_step.n_iter, 1);
    assert_eq!(one_step.termination_reason, "max_iter_reached");
    assert!(one_step.final_loglik_change.is_finite());
    assert!(one_step.final_relative_loglik_change.is_finite());
}

/// Literature-grade Monte-Carlo (>=500 reps): higher-order G-DINA recovery of the
/// saturated item probabilities and the higher-order parameters under a normal and
/// a skewed (mis-specified prior) trait distribution.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_ho_gdina_recovery_500() {
    let (n_attr, n_items, n, reps) = (3usize, 15usize, 1500usize, 500usize);
    let q = hogdina_q3();
    let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
    // additive saturated truth: p_il = 0.15 + 0.7 * popcount(l)/K_i
    let mut truth = vec![0.0f64; item_off[n_items]];
    for i in 0..n_items {
        let (a0, b0) = (item_off[i], item_off[i + 1]);
        for (li, l) in (a0..b0).enumerate() {
            truth[l] = 0.15 + 0.7 * (li.count_ones() as f64) / kreq[i] as f64;
        }
    }
    let a_true = vec![1.2f64, 1.5, 0.9];
    let d_true = vec![0.3f64, -0.5, 0.6];
    for &skew in [false, true].iter() {
        let (mut wp, mut ra, mut attr, mut nconv) = (0.0f64, 0.0f64, 0.0f64, 0usize);
        for rep in 0..reps {
            let mut rng = Lcg(0x27BB2EE687B0B0FDu64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15));
            let (y, profiles, _t) = simulate_ho_gdina(
                &a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, skew, &mut rng,
            );
            let observed = vec![true; n * n_items];
            let res =
                fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
            if res.converged {
                nconv += 1;
            }
            // mass-weighted RMSE(p) so near-empty classes don't dominate
            let mut mass = vec![0.0f64; item_off[n_items]];
            for &c in &profiles {
                for i in 0..n_items {
                    mass[item_off[i] + reduce_class(c, qmask[i])] += 1.0;
                }
            }
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for x in 0..item_off[n_items] {
                let e = res.item_prob[x] - truth[x];
                num += mass[x] * e * e;
                den += mass[x];
            }
            wp += (num / den).sqrt() / reps as f64;
            ra += rmse(&res.attr_slope, &a_true) / reps as f64;
            attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr) / reps as f64;
        }
        println!(
            "[HO-GDINA MC skew={skew}] reps={reps} conv={:.2} wRMSE(p)={:.4} RMSE(a)={:.3} attr-agree={:.3}",
            nconv as f64 / reps as f64,
            wp,
            ra,
            attr
        );
        assert_eq!(
            nconv,
            reps,
            "nonconverged replications: {} of {reps} (skew={skew})",
            reps - nconv
        );
        assert!(wp < 0.04, "wRMSE(p) {wp} skew={skew}");
        assert!(attr > 0.90, "attribute agreement {attr} skew={skew}");
    }
}

// ----- Sequential G-DINA polytomous CDM (Ma & de la Torre, 2016) -----

/// Deterministic anchor A (category-probability identity): step probs [a, b] give
/// P(0)=1-a, P(1)=a(1-b), P(2)=a*b, summing to 1 — catches a product-direction or
/// trailing-factor (sentinel) off-by-one with no Monte-Carlo noise.
#[test]
fn seq_category_probs_matches_identity() {
    let (a, b) = (0.7, 0.3); // a != b, both != 0.5 (non-centered)
    let p = seq_category_probs(&[a, b]);
    assert!((p[0] - (1.0 - a)).abs() < 1e-12, "P(0)");
    assert!((p[1] - a * (1.0 - b)).abs() < 1e-12, "P(1)");
    assert!(
        (p[2] - a * b).abs() < 1e-12,
        "P(2) top has no trailing factor"
    );
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12, "sum to 1");
    // M=1 collapses to Bernoulli.
    let p1 = seq_category_probs(&[0.8]);
    assert!((p1[0] - 0.2).abs() < 1e-12 && (p1[1] - 0.8).abs() < 1e-12);
    // M=3 telescopes to 1 for an asymmetric table.
    let p3 = seq_category_probs(&[0.6, 0.4, 0.3]);
    assert!((p3.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((p3[3] - 0.6 * 0.4 * 0.3).abs() < 1e-12);
    // The PRODUCTION log transform (used by the estimator's E-step refresh) exp-matches
    // the literal-anchored reference for interior steps — so the two implementations
    // cannot harbour a shared, mutually-hidden bug.
    for steps in [vec![0.7, 0.3], vec![0.6, 0.4, 0.3], vec![0.9]] {
        let mut lp = vec![0.0f64; steps.len() + 1];
        seq_category_logprobs_into(&steps, 1e-9, &mut lp);
        let pr = seq_category_probs(&steps);
        for (a, b) in lp.iter().zip(&pr) {
            assert!((a.exp() - b).abs() < 1e-12, "log transform {a} vs {b}");
        }
    }
}

/// Deterministic anchor B (at-risk / advanced counts): responses {0,1,1,2} in one
/// reduced class give I=[4,3], R=[3,1], so s_1=3/4, s_2=1/3 — nails the {>=k}/{>=k-1}
/// denominator subsetting that a fit/RMSE test cannot reliably expose.
#[test]
fn seq_scatter_counts_at_risk_denominator() {
    let mut ii = vec![0.0f64; 2];
    let mut rr = vec![0.0f64; 2];
    for &x in &[0usize, 1, 1, 2] {
        seq_scatter_counts(x, 1.0, 2, &mut ii, &mut rr);
    }
    assert_eq!(ii, vec![4.0, 3.0]); // at risk: step1 (x>=0)=4, step2 (x>=1)=3
    assert_eq!(rr, vec![3.0, 1.0]); // advanced: step1 (x>=1)=3, step2 (x>=2)=1
    assert!((rr[0] / ii[0] - 0.75).abs() < 1e-12); // s_1 = 3/4
    assert!((rr[1] / ii[1] - 1.0 / 3.0).abs() < 1e-12); // s_2 = 1/3
}

/// Binary data (M_i = 1 for every item) reduces the sequential G-DINA to fit_gdina
/// BIT-FOR-BIT: identical monotone init, identical E-step logprobs (ln s / ln(1-s)),
/// identical closed-form ratio, so the whole loglik trace and the step/success probs
/// agree to machine precision.
#[test]
fn seq_gdina_reduces_to_gdina_at_m1() {
    let (q, n_items) = wald_q2(3, 3);
    let n = 800usize;
    let (item_off, qmask, truth) = wald_truth(&q, n_items, "acdm");
    let mut rng = Lcg(424242);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = CdmConfig::default();
    let g = fit_gdina(&y, &observed, &q, n, n_items, 2, &cfg).unwrap();
    let sq = fit_seq_gdina(&y, &observed, &q, n, n_items, 2, &cfg).unwrap();
    assert_eq!(
        sq.max_cat,
        vec![1u32; n_items],
        "all items binary -> M_i = 1"
    );
    assert_eq!(sq.step_prob.len(), g.item_prob.len());
    assert_eq!(
        sq.loglik_trace.len(),
        g.loglik_trace.len(),
        "same iteration count"
    );
    assert_eq!(sq.n_iter, g.n_iter);
    assert_eq!(sq.converged, g.converged);
    for (a, b) in sq.loglik_trace.iter().zip(&g.loglik_trace) {
        assert!((a - b).abs() < 1e-12, "loglik trace {a} vs {b}");
    }
    for (a, b) in sq.step_prob.iter().zip(&g.item_prob) {
        assert!((a - b).abs() < 1e-12, "step prob {a} vs {b}");
    }
    // P(X=1|l) == fit_gdina p_il, P(X=0|l) == 1 - p_il.
    for i in 0..n_items {
        let rw = 1usize << sq.k_required[i];
        for l in 0..rw {
            let p1 = sq.cat_prob[sq.cat_off[i] + l * 2 + 1];
            let p0 = sq.cat_prob[sq.cat_off[i] + l * 2];
            let pg = g.item_prob[g.item_off[i] + l];
            assert!((p1 - pg).abs() < 1e-12 && (p0 - (1.0 - pg)).abs() < 1e-12);
        }
    }
}

/// Draw ordered polytomous responses from per-item, per-class step tables, using the
/// SAME class-major reduce_class layout the estimator recovers (spec-fix: matched
/// classes). Sequential draw: advance while Bernoulli(s_k) succeeds, stop at first fail.
fn simulate_seq_gdina(
    qmask: &[usize],
    s_off: &[usize],
    max_cat: &[u32],
    truth_steps: &[f64],
    profiles: &[usize],
    n_items: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let n = profiles.len();
    let mut y = vec![0.0f64; n * n_items];
    for j in 0..n {
        for i in 0..n_items {
            let m = max_cat[i] as usize;
            let l = reduce_class(profiles[j], qmask[i]);
            let base = s_off[i] + l * m;
            let mut cat = 0usize;
            for k in 1..=m {
                if rng.next_f64() < truth_steps[base + (k - 1)] {
                    cat = k;
                } else {
                    break;
                }
            }
            y[j * n_items + i] = cat as f64;
        }
    }
    y
}

/// K=2 design: `n_single` single-attribute M=1 items per attribute (identification) +
/// `n_pair` two-attribute M=2 polytomous items with an ASYMMETRIC, mastery-increasing
/// step table. Returns (q, qmask, s_off, max_cat, truth_steps).
#[allow(clippy::type_complexity)]
fn seq_design(
    n_single: usize,
    n_pair: usize,
) -> (Vec<u8>, Vec<usize>, Vec<usize>, Vec<u32>, Vec<f64>) {
    let k = 2usize;
    let mut q: Vec<u8> = Vec::new();
    for _ in 0..n_single {
        q.extend_from_slice(&[1, 0]);
    }
    for _ in 0..n_single {
        q.extend_from_slice(&[0, 1]);
    }
    for _ in 0..n_pair {
        q.extend_from_slice(&[1, 1]);
    }
    let n_items = 2 * n_single + n_pair;
    let mut qmask = vec![0usize; n_items];
    let mut kreq = vec![0u32; n_items];
    for i in 0..n_items {
        qmask[i] = qmask_of(&q, i, k);
        kreq[i] = qmask[i].count_ones();
    }
    let mut max_cat = vec![1u32; n_items];
    for m in max_cat.iter_mut().skip(2 * n_single) {
        *m = 2;
    }
    let mut s_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        s_off[i + 1] = s_off[i] + (max_cat[i] as usize) * (1usize << kreq[i]);
    }
    let mut truth = vec![0.0f64; s_off[n_items]];
    for i in 0..(2 * n_single) {
        // M=1, K=1: [non-master, master]
        truth[s_off[i]] = 0.20;
        truth[s_off[i] + 1] = 0.85;
    }
    // M=2, K=2, class-major [l*2 + (k-1)]; asymmetric (s1 != s2), mastery-increasing.
    let pair = [[0.25, 0.15], [0.55, 0.30], [0.50, 0.25], [0.85, 0.70]];
    for i in (2 * n_single)..n_items {
        let base = s_off[i];
        for (l, row) in pair.iter().enumerate() {
            truth[base + l * 2] = row[0];
            truth[base + l * 2 + 1] = row[1];
        }
    }
    (q, qmask, s_off, max_cat, truth)
}

/// Non-trivial ordered recovery: fit the shared-Q sequential G-DINA on M=2 polytomous
/// data with distinct, asymmetric per-class step tables and recover the step and
/// category probabilities plus attribute classification.
#[test]
fn seq_gdina_recovers_polytomous_steps() {
    let k = 2usize;
    let (n_single, n_pair) = (5usize, 5usize);
    let (q, qmask, s_off, max_cat, truth) = seq_design(n_single, n_pair);
    let n_items = 2 * n_single + n_pair;
    let n = 5000usize;
    let mut rng = Lcg(20160716);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate_seq_gdina(
        &qmask, &s_off, &max_cat, &truth, &profiles, n_items, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_seq_gdina(&y, &observed, &q, n, n_items, k, &CdmConfig::default()).unwrap();
    assert_eq!(res.max_cat, max_cat, "derived max categories");
    assert_eq!(res.s_off, s_off, "step layout");
    let rm = rmse(&res.step_prob, &truth);
    assert!(rm < 0.05, "step-prob RMSE {rm}");
    // Category-prob recovery for the pair items (the stable, PRIMARY quantity).
    let pair = [[0.25, 0.15], [0.55, 0.30], [0.50, 0.25], [0.85, 0.70]];
    for i in (2 * n_single)..n_items {
        let m1 = max_cat[i] as usize + 1;
        for (l, row) in pair.iter().enumerate() {
            let tc = seq_category_probs(row);
            for (x, &tcx) in tc.iter().enumerate().take(m1) {
                let est = res.cat_prob[res.cat_off[i] + l * m1 + x];
                assert!(
                    (est - tcx).abs() < 0.04,
                    "cat i{i} l{l} x{x}: {est} vs {tcx}"
                );
            }
        }
    }
    // Attribute classification agreement.
    let mut correct = 0usize;
    for j in 0..n {
        for kk in 0..k {
            let est = (res.attr_prob[j * k + kk] >= 0.5) as usize;
            if est == ((profiles[j] >> kk) & 1) {
                correct += 1;
            }
        }
    }
    let acc = correct as f64 / (n * k) as f64;
    assert!(acc > 0.85, "attribute accuracy {acc}");
}

/// Missing (MAR) is dropped; malformed input is rejected — including the sequential
/// pitfall of an item stuck at category 0 (measures nothing), while a zero-frequency
/// INTERIOR category is accepted (legitimate under a continuation-ratio model).
#[test]
fn seq_gdina_handles_missing_and_validates() {
    let k = 2usize;
    let (n_single, n_pair) = (3usize, 2usize);
    let (q, qmask, s_off, max_cat, truth) = seq_design(n_single, n_pair);
    let n_items = 2 * n_single + n_pair;
    let n = 400usize;
    let mut rng = Lcg(77);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate_seq_gdina(
        &qmask, &s_off, &max_cat, &truth, &profiles, n_items, &mut rng,
    );
    let cfg = CdmConfig::default();
    // Valid fit with a few missing cells.
    let mut observed = vec![true; n * n_items];
    observed[0] = false;
    observed[n_items + 1] = false;
    let res = fit_seq_gdina(&y, &observed, &q, n, n_items, k, &cfg).unwrap();
    assert!(!res.loglik_trace.is_empty());
    assert!(
        res.converged,
        "termination={} n_iter={} delta={} tolerance={}",
        res.termination_reason, res.n_iter, res.final_loglik_change, res.stopping_tolerance
    );
    assert_eq!(res.termination_reason, "tolerance_met");
    assert!(res.final_loglik_change.abs() < res.stopping_tolerance);
    assert!(res.final_relative_loglik_change.is_finite());
    let all_obs = vec![true; n * n_items];
    // Non-integer category.
    let mut ybad = y.clone();
    ybad[10] = 1.5;
    assert!(fit_seq_gdina(&ybad, &all_obs, &q, n, n_items, k, &cfg).is_err());
    // Negative category.
    let mut yneg = y.clone();
    yneg[10] = -1.0;
    assert!(fit_seq_gdina(&yneg, &all_obs, &q, n, n_items, k, &cfg).is_err());
    // A pair item stuck at category 0 (never leaves 0) -> rejected.
    let mut yzero = y.clone();
    for j in 0..n {
        yzero[j * n_items + 2 * n_single] = 0.0;
    }
    assert!(fit_seq_gdina(&yzero, &all_obs, &q, n, n_items, k, &cfg).is_err());
    // Shape mismatch.
    assert!(fit_seq_gdina(&y[..y.len() - 1], &all_obs, &q, n, n_items, k, &cfg).is_err());
    // A zero-frequency INTERIOR category must NOT be rejected: force item (2*n_single)
    // to skip category 1 (only 0 and 2 observed) — still a valid sequential item.
    let mut yskip = y.clone();
    let it = 2 * n_single;
    for j in 0..n {
        let v = yskip[j * n_items + it];
        yskip[j * n_items + it] = if v >= 1.0 { 2.0 } else { 0.0 };
    }
    // max observed category is 2 (some persons reach 2), interior cat 1 has 0 freq.
    assert!(fit_seq_gdina(&yskip, &all_obs, &q, n, n_items, k, &cfg).is_ok());

    // Iteration-limited fits expose exact nonconvergence evidence instead of requiring
    // callers to infer the reason and stopping metric from the likelihood trace.
    let one_cfg = CdmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..CdmConfig::default()
    };
    let one = fit_seq_gdina(&y, &all_obs, &q, n, n_items, k, &one_cfg).unwrap();
    assert!(!one.converged);
    assert_eq!(one.n_iter, 1);
    assert_eq!(one.termination_reason, "max_iter_reached");
    assert!(one.final_loglik_change.is_finite());
    assert!(one.final_relative_loglik_change.is_finite());
    assert_eq!(one.stopping_tolerance, one_cfg.tol);
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the sequential G-DINA step and
/// category probabilities under BOTH a normal and a right-skew higher-order attribute
/// distribution (fitting a free pi_c). Primary hard assertion is the category-prob
/// RMSE (the stable, model-predicted quantity); the step RMSE is weighted by realized
/// AT-RISK mass (top steps are inherently noisier) and reported as secondary.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_seq_gdina_recovery_500() {
    let reps = 500usize;
    let k = 3usize;
    let n = 2500usize;
    // 3 single M=1 items per attribute (identification) + M=2 and M=3 polytomous items.
    let mut q: Vec<u8> = Vec::new();
    for a in 0..k {
        for _ in 0..3 {
            let mut r = vec![0u8; k];
            r[a] = 1;
            q.extend_from_slice(&r);
        }
    }
    // polytomous items on attribute pairs/triples.
    let poly_q: [&[usize]; 4] = [&[0, 1], &[0, 2], &[1, 2], &[0, 1, 2]];
    let poly_m: [u32; 4] = [2, 2, 3, 3]; // include M=3 (>=2 interior steps)
    for pq in poly_q.iter() {
        let mut r = vec![0u8; k];
        for &a in pq.iter() {
            r[a] = 1;
        }
        q.extend_from_slice(&r);
    }
    let n_items = 3 * k + poly_q.len();
    let mut qmask = vec![0usize; n_items];
    let mut kreq = vec![0u32; n_items];
    for i in 0..n_items {
        qmask[i] = qmask_of(&q, i, k);
        kreq[i] = qmask[i].count_ones();
    }
    let mut max_cat = vec![1u32; n_items];
    for (j, &m) in poly_m.iter().enumerate() {
        max_cat[3 * k + j] = m;
    }
    let mut s_off = vec![0usize; n_items + 1];
    let mut cat_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        s_off[i + 1] = s_off[i] + (max_cat[i] as usize) * (1usize << kreq[i]);
        cat_off[i + 1] = cat_off[i] + (max_cat[i] as usize + 1) * (1usize << kreq[i]);
    }
    // Truth step tables: mastery-increasing (more mastered required attrs -> higher
    // continuation at every step), step decreasing in k (higher categories harder).
    let mut truth = vec![0.0f64; s_off[n_items]];
    for i in 0..n_items {
        let m = max_cat[i] as usize;
        let rw = 1usize << kreq[i];
        let ki = kreq[i] as f64;
        for l in 0..rw {
            let frac = l.count_ones() as f64 / ki; // fraction of required attrs mastered
            for kk in 0..m {
                // step 1 base ~0.30..0.90; each higher step -0.12; +mastery.
                let base = 0.30 + 0.55 * frac - 0.12 * kk as f64;
                truth[s_off[i] + l * m + kk] = base.clamp(0.08, 0.92);
            }
        }
    }
    // Strong single-attribute M=1 identification items (guess 0.12, mastery 0.90) so
    // the profile posterior is sharp; the polytomous items carry the recovery target.
    for i in 0..(3 * k) {
        truth[s_off[i]] = 0.12;
        truth[s_off[i] + 1] = 0.90;
    }
    // Higher-order attribute parameters (2PL): theta -> mastery.
    let a_ho = vec![1.2f64; k];
    let d_ho: Vec<f64> = (0..k).map(|kk| 0.4 - 0.4 * kk as f64).collect();

    for &skew in [false, true].iter() {
        let (mut wnum, mut wden) = (0.0f64, 0.0f64);
        let (mut cat_se, mut cat_cells) = (0.0f64, 0.0f64);
        let (mut attr_ok, mut attr_tot) = (0.0f64, 0.0f64);
        let mut nconv = 0usize;
        let mut min_atrisk = f64::INFINITY;
        for rep in 0..reps {
            let mut rng = Lcg(0x9E3779B97F4A7C15u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03));
            let profiles: Vec<usize> = (0..n)
                .map(|_| {
                    let theta = if skew {
                        // standardized shifted chi-square(3): mean 0, var 1, right-skew.
                        let mut cc = 0.0;
                        for _ in 0..3 {
                            let z = rng.normal();
                            cc += z * z;
                        }
                        (cc - 3.0) / 6.0_f64.sqrt()
                    } else {
                        rng.normal()
                    };
                    let mut c = 0usize;
                    for kk in 0..k {
                        let p = 1.0 / (1.0 + (-(a_ho[kk] * theta + d_ho[kk])).exp());
                        if rng.next_f64() < p {
                            c |= 1 << kk;
                        }
                    }
                    c
                })
                .collect();
            let y = simulate_seq_gdina(
                &qmask, &s_off, &max_cat, &truth, &profiles, n_items, &mut rng,
            );
            let observed = vec![true; n * n_items];
            let res =
                fit_seq_gdina(&y, &observed, &q, n, n_items, k, &CdmConfig::default()).unwrap();
            assert!(
                res.converged,
                "rep {rep} skew={skew}: termination={} n_iter={} delta={} relative_delta={} tolerance={}",
                res.termination_reason,
                res.n_iter,
                res.final_loglik_change,
                res.final_relative_loglik_change,
                res.stopping_tolerance
            );
            assert_eq!(res.termination_reason, "tolerance_met");
            assert!(res.final_loglik_change.abs() < res.stopping_tolerance);
            nconv += 1;
            assert_eq!(
                res.max_cat, max_cat,
                "derived M_i matches design (rep {rep})"
            );
            // Invariants: every step/category prob finite in (0,1); category probs sum to 1.
            for &sp in &res.step_prob {
                assert!(sp.is_finite() && sp > 0.0 && sp < 1.0, "step prob {sp}");
            }
            for i in 0..n_items {
                let m1 = max_cat[i] as usize + 1;
                let rw = 1usize << kreq[i];
                for l in 0..rw {
                    let mut s = 0.0;
                    for x in 0..m1 {
                        let p = res.cat_prob[res.cat_off[i] + l * m1 + x];
                        assert!(p.is_finite() && p >= 0.0, "cat prob {p}");
                        s += p;
                    }
                    assert!((s - 1.0).abs() < 1e-9, "category simplex {s}");
                }
            }
            // Realized at-risk mass I_ik(l) from true profiles, for step weighting.
            let mut atrisk = vec![0.0f64; s_off[n_items]];
            let mut advanced = vec![0.0f64; s_off[n_items]];
            for j in 0..n {
                for i in 0..n_items {
                    let m = max_cat[i] as usize;
                    let l = reduce_class(profiles[j], qmask[i]);
                    let base = s_off[i] + l * m;
                    let x = y[j * n_items + i] as usize;
                    seq_scatter_counts(
                        x,
                        1.0,
                        m,
                        &mut atrisk[base..base + m],
                        &mut advanced[base..base + m],
                    );
                }
            }
            for cell in 0..s_off[n_items] {
                let w = atrisk[cell];
                if w > 0.0 {
                    min_atrisk = min_atrisk.min(w);
                    let e = res.step_prob[cell] - truth[cell];
                    wnum += w * e * e;
                    wden += w;
                }
            }
            // Category-prob RMSE vs the model-implied truth (primary, stable).
            for i in 0..n_items {
                let m = max_cat[i] as usize;
                let m1 = m + 1;
                let rw = 1usize << kreq[i];
                for l in 0..rw {
                    let tsteps = &truth[s_off[i] + l * m..s_off[i] + l * m + m];
                    let tc = seq_category_probs(tsteps);
                    for x in 0..m1 {
                        let e = res.cat_prob[res.cat_off[i] + l * m1 + x] - tc[x];
                        cat_se += e * e;
                        cat_cells += 1.0;
                    }
                }
            }
            for j in 0..n {
                for kk in 0..k {
                    let est = (res.attr_prob[j * k + kk] >= 0.5) as usize;
                    if est == ((profiles[j] >> kk) & 1) {
                        attr_ok += 1.0;
                    }
                    attr_tot += 1.0;
                }
            }
        }
        let wrmse_step = (wnum / wden).sqrt();
        let rmse_cat = (cat_se / cat_cells).sqrt();
        let attr = attr_ok / attr_tot;
        let conv = nconv as f64 / reps as f64;
        println!(
            "[seq-gdina MC skew={skew}] reps={reps} conv={conv:.3} \
             wRMSE(step|at-risk)={wrmse_step:.4} RMSE(cat)={rmse_cat:.4} \
             attr={attr:.3} min_at_risk_mass={min_atrisk:.1}"
        );
        // Category probs are the stable primary target; step probs (esp. top steps
        // starved under skew, min at-risk mass reported above) are looser and
        // at-risk-weighted. Thresholds calibrated to this K=3, M in {2,3} design.
        assert!(rmse_cat < 0.03, "category-prob RMSE {rmse_cat} skew={skew}");
        assert!(
            wrmse_step < 0.05,
            "at-risk-weighted step RMSE {wrmse_step} skew={skew}"
        );
        assert!(attr > 0.92, "attribute agreement {attr} skew={skew}");
        assert_eq!(nconv, reps, "every calibration must converge skew={skew}");
    }
}

// ----- Per-step-Q sequential G-DINA (Ma & de la Torre, 2016, restricted-Q) -----

/// Simulate per-step-Q sequential responses: step v of item i succeeds with probability
/// `step_truth[step_off[i]+v-1][reduce_class(profile, step_qmask[g])]`.
fn simulate_seq_gdina_qr(
    step_off: &[usize],
    step_qmask: &[usize],
    spo_kq: &[u32],     // |q_ik| per step row (for the truth table width)
    step_truth: &[f64], // step-row-major, spo-indexed
    spo: &[usize],
    n_steps: &[usize],
    profiles: &[usize],
    n_items: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let _ = spo_kq;
    let n = profiles.len();
    let mut y = vec![0.0f64; n * n_items];
    for j in 0..n {
        for i in 0..n_items {
            let m = n_steps[i];
            let mut cat = 0usize;
            for v in 1..=m {
                let g = step_off[i] + (v - 1);
                let l = reduce_class(profiles[j], step_qmask[g]);
                if rng.next_f64() < step_truth[spo[g] + l] {
                    cat = v;
                } else {
                    break;
                }
            }
            y[j * n_items + i] = cat as f64;
        }
    }
    y
}

/// Shared-Q reduction: with every step of an item sharing the item's Q, fit_seq_gdina_qr
/// matches the shipped shared-Q fit_seq_gdina. loglik and cat_prob zip bit-exactly; step_prob
/// is compared CELL-BY-CELL through the transposed layout map (class-major vs step-row-major).
#[test]
fn seq_gdina_qr_reduces_to_shared_q() {
    let (q, qmask, s_off_t, max_cat_t, truth) = seq_design(4, 4);
    let n_items = 2 * 4 + 4;
    let n = 3000usize;
    let mut rng = Lcg(20240101);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << 2)).collect();
    let y = simulate_seq_gdina(
        &qmask, &s_off_t, &max_cat_t, &truth, &profiles, n_items, &mut rng,
    );
    let observed = vec![true; n * n_items];
    let cfg = CdmConfig::default();
    let shared = fit_seq_gdina(&y, &observed, &q, n, n_items, 2, &cfg).unwrap();
    let n_steps: Vec<usize> = shared.max_cat.iter().map(|&m| m as usize).collect();
    let mut step_q: Vec<u8> = Vec::new();
    for i in 0..n_items {
        for _ in 0..n_steps[i] {
            step_q.extend_from_slice(&q[i * 2..i * 2 + 2]);
        }
    }
    let qr = fit_seq_gdina_qr(&y, &observed, &step_q, &n_steps, n, n_items, 2, &cfg).unwrap();
    assert_eq!(qr.loglik_trace.len(), shared.loglik_trace.len());
    for (a, b) in qr.loglik_trace.iter().zip(&shared.loglik_trace) {
        assert!((a - b).abs() < 1e-12, "loglik {a} vs {b}");
    }
    for (a, b) in qr.cat_prob.iter().zip(&shared.cat_prob) {
        assert!((a - b).abs() < 1e-12, "cat_prob {a} vs {b}");
    }
    assert_eq!(qr.n_parameters, shared.n_parameters);
    // step_prob: shared s_off[i]+l*M+(k-1) (class-major) vs qr spo[step_off[i]+(k-1)]+l.
    for i in 0..n_items {
        let m = shared.max_cat[i] as usize;
        let rw = 1usize << shared.k_required[i];
        for l in 0..rw {
            for k in 1..=m {
                let sh = shared.step_prob[shared.s_off[i] + l * m + (k - 1)];
                let g = qr.step_off[i] + (k - 1);
                let qv = qr.step_prob[qr.spo[g] + l];
                assert!((sh - qv).abs() < 1e-12, "step i{i} l{l} k{k}: {sh} vs {qv}");
            }
        }
    }
}

/// Non-trivial STEP-DISTINCT recovery with a NON-CONTIGUOUS union: an item whose step 1
/// requires attribute 0 only and step 2 requires attributes {0, 2} (union {0,2} is
/// non-contiguous — a naive union-mask-AND derivation would misread the step class). Asserts
/// (a) per-step block WIDTHS (2 and 4 — the only thing that catches over-collapse), (b) a
/// large B-contrast in step 2 is recovered (gap >= 0.4), and (c) step 1 is flat in attr 2.
#[test]
fn seq_gdina_qr_recovers_step_distinct() {
    let k = 3usize; // attrs 0,1,2
                    // items: 3 single-attr M=1 identification items per attribute (pins each dim) + 1
                    // step-distinct M=2 item (step1 q={0}, step2 q={0,2}).
    let mut step_q: Vec<u8> = Vec::new();
    let mut n_steps: Vec<usize> = Vec::new();
    for a in 0..k {
        for _ in 0..3 {
            let mut r = vec![0u8; k];
            r[a] = 1;
            step_q.extend_from_slice(&r); // one step row
            n_steps.push(1);
        }
    }
    // the step-distinct item: step1 {0}, step2 {0,2}
    step_q.extend_from_slice(&[1, 0, 0]); // step 1 q = {0}
    step_q.extend_from_slice(&[1, 0, 1]); // step 2 q = {0,2}
    n_steps.push(2);
    let n_items = 3 * k + 1;
    let sd = n_items - 1; // the step-distinct item index

    // truth: singles guess 0.15 / master 0.90; step-distinct item step1 (q={0}: classes
    // [a0=0,a0=1]) = [0.30, 0.80]; step2 (q={0,2}: classes [00,10,01,11] over (a0,a2)) with a
    // LARGE a2-contrast: s2(a0=1,a2=0)=0.20 vs s2(a0=1,a2=1)=0.80.
    // Build step_off/spo/step_qmask to drive the simulator.
    let mut step_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        step_off[i + 1] = step_off[i] + n_steps[i];
    }
    let n_rows = step_off[n_items];
    let mut step_qmask = vec![0usize; n_rows];
    let mut spo = vec![0usize; n_rows + 1];
    for g in 0..n_rows {
        let mut m = 0usize;
        for a in 0..k {
            if step_q[g * k + a] != 0 {
                m |= 1 << a;
            }
        }
        step_qmask[g] = m;
        spo[g + 1] = spo[g] + (1usize << m.count_ones());
    }
    let mut truth = vec![0.0f64; spo[n_rows]];
    for i in 0..(3 * k) {
        // single M=1 identification items (K=1: classes [non,master])
        truth[spo[step_off[i]]] = 0.15;
        truth[spo[step_off[i]] + 1] = 0.90;
    }
    // step-distinct item
    let g1 = step_off[sd]; // step 1, q={0}: classes [a0=0, a0=1]
    truth[spo[g1]] = 0.30;
    truth[spo[g1] + 1] = 0.80;
    let g2 = step_off[sd] + 1; // step 2, q={0,2}: reduce_class over {0,2} = a0 + 2*a2
    truth[spo[g2]] = 0.15; // (a0=0,a2=0)
    truth[spo[g2] + 1] = 0.20; // (a0=1,a2=0)
    truth[spo[g2] + 2] = 0.20; // (a0=0,a2=1)
    truth[spo[g2] + 3] = 0.80; // (a0=1,a2=1)  <- large a2 contrast at a0=1

    let n = 6000usize;
    let mut rng = Lcg(916);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate_seq_gdina_qr(
        &step_off,
        &step_qmask,
        &[],
        &truth,
        &spo,
        &n_steps,
        &profiles,
        n_items,
        &mut rng,
    );
    let observed = vec![true; n * n_items];
    let res = fit_seq_gdina_qr(
        &y,
        &observed,
        &step_q,
        &n_steps,
        n,
        n_items,
        k,
        &CdmConfig::default(),
    )
    .unwrap();
    assert!(res.converged);
    // (a) STRUCTURE: the step-distinct item's step blocks have widths 2 and 4.
    let g1r = res.step_off[sd];
    let g2r = res.step_off[sd] + 1;
    assert_eq!(
        res.spo[g1r + 1] - res.spo[g1r],
        2,
        "step 1 width = 2^{{|q1|}}"
    );
    assert_eq!(
        res.spo[g2r + 1] - res.spo[g2r],
        4,
        "step 2 width = 2^{{|q2|}}"
    );
    assert_eq!(res.step_kq[g1r], 1);
    assert_eq!(res.step_kq[g2r], 2);
    // n_parameters reflects the per-step widths (2 + 4 for the step-distinct item).
    let total_step_params: usize = (0..n_rows).map(|g| res.spo[g + 1] - res.spo[g]).sum();
    assert_eq!(res.n_parameters, total_step_params + ((1 << k) - 1));
    // (b) large a2-contrast in step 2 recovered (gap >= 0.4).
    let s2_a1_b0 = res.step_prob[res.spo[g2r] + 1]; // (a0=1,a2=0)
    let s2_a1_b1 = res.step_prob[res.spo[g2r] + 3]; // (a0=1,a2=1)
    assert!(
        s2_a1_b1 - s2_a1_b0 > 0.4,
        "step-2 a2 contrast {s2_a1_b0} -> {s2_a1_b1}"
    );
    // (c) step 1 is (near) flat in attr 2 (it only depends on a0): both a0=1 draws equal.
    // step 1 has only 2 classes (a0), so it is structurally flat in a2 by construction; assert
    // the recovered step-1 master prob is near 0.80 and non-master near 0.30.
    assert!(
        (res.step_prob[res.spo[g1r]] - 0.30).abs() < 0.06,
        "step1 non-master"
    );
    assert!(
        (res.step_prob[res.spo[g1r] + 1] - 0.80).abs() < 0.06,
        "step1 master"
    );
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "EM monotone");
    }
}

#[test]
fn seq_gdina_qr_validates() {
    let k = 2usize;
    // valid: 2 single items + 1 M=2 step-distinct-ish item (step1 {0}, step2 {0,1})
    let mut step_q: Vec<u8> = vec![1, 0, /*item0 step1*/ 0, 1 /*item1 step1*/];
    let mut n_steps = vec![1usize, 1];
    step_q.extend_from_slice(&[1, 0]); // item2 step1 {0}
    step_q.extend_from_slice(&[1, 1]); // item2 step2 {0,1}
    n_steps.push(2);
    let n_items = 3usize;
    let n = 300usize;
    // build a simple valid y via the simulator
    let mut step_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        step_off[i + 1] = step_off[i] + n_steps[i];
    }
    let n_rows = step_off[n_items];
    let mut step_qmask = vec![0usize; n_rows];
    let mut spo = vec![0usize; n_rows + 1];
    for g in 0..n_rows {
        let mut m = 0usize;
        for a in 0..k {
            if step_q[g * k + a] != 0 {
                m |= 1 << a;
            }
        }
        step_qmask[g] = m;
        spo[g + 1] = spo[g] + (1usize << m.count_ones());
    }
    let mut truth = vec![0.5f64; spo[n_rows]];
    truth[spo[step_off[0]]] = 0.2;
    truth[spo[step_off[0]] + 1] = 0.85;
    truth[spo[step_off[1]]] = 0.2;
    truth[spo[step_off[1]] + 1] = 0.85;
    let mut rng = Lcg(3);
    let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
    let y = simulate_seq_gdina_qr(
        &step_off,
        &step_qmask,
        &[],
        &truth,
        &spo,
        &n_steps,
        &profiles,
        n_items,
        &mut rng,
    );
    let cfg = CdmConfig::default();
    let obs = vec![true; n * n_items];
    // valid fit (if item2 reaches category 2 for someone; make sure the design does)
    let ok = fit_seq_gdina_qr(&y, &obs, &step_q, &n_steps, n, n_items, k, &cfg);
    assert!(ok.is_ok(), "valid: {:?}", ok.err());
    let mut missing = obs.clone();
    missing[0] = false;
    assert!(fit_seq_gdina_qr(&y, &missing, &step_q, &n_steps, n, n_items, k, &cfg).is_ok());
    // n_steps length mismatch
    assert!(fit_seq_gdina_qr(&y, &obs, &step_q, &n_steps[..2], n, n_items, k, &cfg).is_err());
    // all-zero step-q row (a step measuring nothing)
    let mut zq = step_q.clone();
    zq[0] = 0; // item0 step1 was {0} -> now all-zero
    assert!(fit_seq_gdina_qr(&y, &obs, &zq, &n_steps, n, n_items, k, &cfg).is_err());
    // all-zero COLUMN: an attribute required by no step. ISOLATE this guard from the
    // all-zero-ROW guard that precedes it by keeping every row non-empty -- two items whose
    // only step is {0}, so attr1 appears in no column while no row is all-zero (a naive
    // fixture that empties attr1's only single-attr step trips the row guard first and would
    // let a deletion of the column guard survive).
    let col_q: Vec<u8> = vec![1, 0, 1, 0];
    let col_ns = vec![1usize, 1];
    let col_y = vec![0.0f64; n * 2];
    let col_obs = vec![true; n * 2];
    let col_err = fit_seq_gdina_qr(&col_y, &col_obs, &col_q, &col_ns, n, 2, k, &cfg).unwrap_err();
    assert!(
        col_err.contains("required by no step"),
        "expected column guard, got: {col_err}"
    );
    // max observed category != declared n_steps: clamp item2 (declared M=2) so its data never
    // reaches category 2. sum(n_steps)=4 still matches the 4 step_q rows, so the length guard
    // passes and the max-observed guard is what must reject it (else x = y as usize could
    // exceed M_i and index clp past the item's (M_i+1)-wide block).
    let mut y_low = y.clone();
    for p in 0..n {
        let idx = p * n_items + 2;
        if y_low[idx] > 1.0 {
            y_low[idx] = 1.0;
        }
    }
    let low_err =
        fit_seq_gdina_qr(&y_low, &obs, &step_q, &n_steps, n, n_items, k, &cfg).unwrap_err();
    assert!(
        low_err.contains("max observed category"),
        "expected max-observed guard, got: {low_err}"
    );
    // non-integer response
    let mut yb = y.clone();
    yb[5] = 1.5;
    assert!(fit_seq_gdina_qr(&yb, &obs, &step_q, &n_steps, n, n_items, k, &cfg).is_err());
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the per-step-Q sequential G-DINA under
/// normal and skew higher-order attribute distributions.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_seq_gdina_qr_recovery_500() {
    let reps = 500usize;
    let k = 3usize;
    let n = 2000usize;
    // 3 single M=1 items per attribute (identification) + step-distinct polytomous items.
    let mut step_q: Vec<u8> = Vec::new();
    let mut n_steps: Vec<usize> = Vec::new();
    for a in 0..k {
        for _ in 0..3 {
            let mut r = vec![0u8; k];
            r[a] = 1;
            step_q.extend_from_slice(&r);
            n_steps.push(1);
        }
    }
    // step-distinct items: (step1 {0}, step2 {0,1}); (step1 {1}, step2 {1,2}); (step1 {2},
    // step2 {0,2}, step3 {0,1,2}).
    let poly: [&[&[usize]]; 3] = [
        &[&[0], &[0, 1]],
        &[&[1], &[1, 2]],
        &[&[2], &[0, 2], &[0, 1, 2]],
    ];
    for steps in poly.iter() {
        for stp in steps.iter() {
            let mut r = vec![0u8; k];
            for &a in stp.iter() {
                r[a] = 1;
            }
            step_q.extend_from_slice(&r);
        }
        n_steps.push(steps.len());
    }
    let n_items = 3 * k + poly.len();
    let mut step_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        step_off[i + 1] = step_off[i] + n_steps[i];
    }
    let n_rows = step_off[n_items];
    let mut step_qmask = vec![0usize; n_rows];
    let mut spo = vec![0usize; n_rows + 1];
    for g in 0..n_rows {
        let mut m = 0usize;
        for a in 0..k {
            if step_q[g * k + a] != 0 {
                m |= 1 << a;
            }
        }
        step_qmask[g] = m;
        spo[g + 1] = spo[g] + (1usize << m.count_ones());
    }
    // truth step tables: mastery-increasing per step (more mastered required attrs -> higher).
    let mut truth = vec![0.0f64; spo[n_rows]];
    for g in 0..n_rows {
        let rw = 1usize << step_qmask[g].count_ones();
        let kq = step_qmask[g].count_ones() as f64;
        for l in 0..rw {
            let frac = l.count_ones() as f64 / kq;
            truth[spo[g] + l] = (0.20 + 0.65 * frac).clamp(0.08, 0.92);
        }
    }
    // strong single identification items
    for i in 0..(3 * k) {
        truth[spo[step_off[i]]] = 0.12;
        truth[spo[step_off[i]] + 1] = 0.90;
    }
    let a_ho = vec![1.2f64; k];
    let d_ho: Vec<f64> = (0..k).map(|kk| 0.4 - 0.4 * kk as f64).collect();

    for &skew in [false, true].iter() {
        let (mut wnum, mut wden) = (0.0f64, 0.0f64);
        let (mut cat_se, mut cat_cnt) = (0.0f64, 0.0f64);
        let (mut attr_ok, mut attr_tot) = (0.0f64, 0.0f64);
        let mut nconv = 0usize;
        for rep in 0..reps {
            let mut rng = Lcg(0x9E3779B97F4A7C15u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03));
            let profiles: Vec<usize> = (0..n)
                .map(|_| {
                    let theta = if skew {
                        let mut cc = 0.0;
                        for _ in 0..3 {
                            let z = rng.normal();
                            cc += z * z;
                        }
                        (cc - 3.0) / 6.0_f64.sqrt()
                    } else {
                        rng.normal()
                    };
                    let mut c = 0usize;
                    for kk in 0..k {
                        let p = 1.0 / (1.0 + (-(a_ho[kk] * theta + d_ho[kk])).exp());
                        if rng.next_f64() < p {
                            c |= 1 << kk;
                        }
                    }
                    c
                })
                .collect();
            let y = simulate_seq_gdina_qr(
                &step_off,
                &step_qmask,
                &[],
                &truth,
                &spo,
                &n_steps,
                &profiles,
                n_items,
                &mut rng,
            );
            let observed = vec![true; n * n_items];
            let res = match fit_seq_gdina_qr(
                &y,
                &observed,
                &step_q,
                &n_steps,
                n,
                n_items,
                k,
                &CdmConfig::default(),
            ) {
                Ok(r) => r,
                Err(_) => continue, // a rep where a poly item did not reach its top category
            };
            if res.converged {
                nconv += 1;
            }
            for w in res.loglik_trace.windows(2) {
                assert!(w[1] >= w[0] - 1e-6, "EM monotone (rep {rep})");
            }
            for &sp in &res.step_prob {
                assert!(sp.is_finite() && sp > 0.0 && sp < 1.0, "step prob {sp}");
            }
            // realized at-risk mass per step cell for weighting.
            let mut atrisk = vec![0.0f64; spo[n_rows]];
            let mut advanced = vec![0.0f64; spo[n_rows]];
            for j in 0..n {
                for i in 0..n_items {
                    let m = n_steps[i];
                    let x = y[j * n_items + i] as usize;
                    for v in 1..=m {
                        let g = step_off[i] + (v - 1);
                        let l = reduce_class(profiles[j], step_qmask[g]);
                        if x >= v - 1 {
                            atrisk[spo[g] + l] += 1.0;
                            if x >= v {
                                advanced[spo[g] + l] += 1.0;
                            }
                        }
                    }
                }
            }
            for cell in 0..spo[n_rows] {
                if atrisk[cell] > 0.0 {
                    let e = res.step_prob[cell] - truth[cell];
                    wnum += atrisk[cell] * e * e;
                    wden += atrisk[cell];
                }
            }
            // category-prob RMSE vs model truth for the poly items.
            for i in (3 * k)..n_items {
                let m = n_steps[i];
                let m1 = m + 1;
                // union class truth: gather step probs per union class via full profiles.
                // compare recovered cat_prob against seq_category_probs of the truth steps
                // at each union class (representative full profile).
                let mut u = 0usize;
                for g in step_off[i]..step_off[i + 1] {
                    u |= step_qmask[g];
                }
                let rwu = 1usize << u.count_ones();
                for c in 0..(1 << k) {
                    let uc = reduce_class(c, u);
                    if uc >= rwu {
                        continue;
                    }
                    let mut steps_t = vec![0.0f64; m];
                    for v in 0..m {
                        let g = step_off[i] + v;
                        steps_t[v] = truth[spo[g] + reduce_class(c, step_qmask[g])];
                    }
                    let tc = seq_category_probs(&steps_t);
                    for x in 0..m1 {
                        let est = res.cat_prob[res.cat_off[i] + uc * m1 + x];
                        let e = est - tc[x];
                        cat_se += e * e;
                        cat_cnt += 1.0;
                    }
                }
            }
            for j in 0..n {
                for kk in 0..k {
                    let est = (res.attr_prob[j * k + kk] >= 0.5) as usize;
                    if est == ((profiles[j] >> kk) & 1) {
                        attr_ok += 1.0;
                    }
                    attr_tot += 1.0;
                }
            }
        }
        let wrmse = (wnum / wden).sqrt();
        let crmse = (cat_se / cat_cnt).sqrt();
        let attr = attr_ok / attr_tot;
        let conv = nconv as f64 / reps as f64;
        println!(
            "[seq-qr MC skew={skew}] reps={reps} conv={conv:.3} wRMSE(step)={wrmse:.4} \
             RMSE(cat)={crmse:.4} attr={attr:.3}"
        );
        assert!(conv > 0.9, "convergence {conv} skew={skew}");
        assert!(crmse < 0.03, "category-prob RMSE {crmse} skew={skew}");
        assert!(
            wrmse < 0.05,
            "at-risk-weighted step RMSE {wrmse} skew={skew}"
        );
        assert!(attr > 0.90, "attribute agreement {attr} skew={skew}");
    }
}
