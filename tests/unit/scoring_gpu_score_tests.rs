use super::*;
use crate::nodes::XiRule;

#[test]
fn gpu_eap_matches_cpu_reduction() {
    let (n_items, n_persons, latent_dim) = (6usize, 40usize, 1usize);
    let alpha: Vec<f64> = (0..n_items).map(|i| 0.1 * i as f64 - 0.2).collect();
    let b: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.2 * i as f64).collect();
    let zeta: Vec<f64> = (0..n_items * latent_dim)
        .map(|i| 0.3 * (i % 3) as f64 - 0.3)
        .collect();
    let fid = vec![0usize; n_items];
    let mut st = 12345u64;
    let mut u = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut y = vec![0.0_f64; n_persons * n_items];
    for v in y.iter_mut() {
        *v = if u() < 0.5 { 1.0 } else { 0.0 };
    }
    let observed = vec![true; n_persons * n_items];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -0.3,
        factor_id: &fid,
        model_type: crate::ModelType::Mls2plm,
        n_dims: 1,
        latent_dim,
        eps_distance: 1e-8,
    };
    let prior = PriorSpec::standard(1);
    let grids = scoring_grids(&bank, 21, XiRule::GaussHermite { q_xi: 11 }).unwrap();
    let ctx = prior_contexts(&prior);
    let config = bank_model_config(&bank, n_persons, n_items);
    let tables = build_tables(
        bank.alpha,
        bank.b,
        bank.zeta,
        bank.tau,
        &config,
        bank.factor_id,
        &ctx,
        &grids,
    );
    let resp = index_responses(&y, &observed, n_persons, n_items);
    let cpu = score_eap_cpu_reduce(&bank, &prior, &grids, &tables, &resp, n_persons, n_items);
    match try_score_eap_gpu(&bank, &prior, &grids, &tables, &resp, n_persons, n_items) {
        None => eprintln!("no GPU adapter present; skipping GPU EAP parity check"),
        Some(gpu) => {
            for p in 0..n_persons {
                assert!(
                    (gpu.loglik[p] - cpu.loglik[p]).abs() < 2e-3,
                    "loglik p={p}: gpu {} vs cpu {}",
                    gpu.loglik[p],
                    cpu.loglik[p]
                );
                assert!((gpu.theta_eap[p] - cpu.theta_eap[p]).abs() < 2e-3);
                assert!((gpu.theta_sd[p] - cpu.theta_sd[p]).abs() < 2e-3);
                assert!((gpu.xi_eap[p] - cpu.xi_eap[p]).abs() < 2e-3);
            }
        }
    }
}
