use super::*;

#[test]
fn test_mask_and_mirt() {
    let params = Params {
        theta: vec![0.0],
        alpha: vec![0.0],
        b: vec![0.0],
        xi: vec![0.0],
        zeta: vec![0.0],
        tau: 0.0,
    };
    let config = ModelConfig {
        n_persons: 1,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-12,
        model_type: ModelType::Mirt,
    };
    let penalty = PenaltyConfig {
        lambda_theta: 0.0,
        lambda_b: 0.0,
        lambda_alpha: 0.0,
        lambda_xi: 0.0,
        lambda_zeta: 0.0,
        lambda_tau: 0.0,
        mu_alpha: 0.0,
        mu_tau: 0.0,
    };
    let y = vec![1.0];
    let mask = vec![false];
    let (obj, _, _) = neg_loglik_and_grad(&y, Some(&mask), &[0], &params, &config, &penalty);
    assert_eq!(obj, 0.0);

    let mask_true = vec![true];
    let (obj_mirt, _, _) =
        neg_loglik_and_grad(&y, Some(&mask_true), &[0], &params, &config, &penalty);
    assert!(obj_mirt > 0.0);
}

#[test]
fn checked_size_arithmetic_reports_overflow() {
    assert_eq!(checked_mul_usize(6, 7, "mul overflow"), Ok(42));
    assert_eq!(
        checked_mul_usize(usize::MAX, 2, "mul overflow"),
        Err("mul overflow".to_owned())
    );
    assert_eq!(checked_add_usize(40, 2, "add overflow"), Ok(42));
    assert_eq!(
        checked_add_usize(usize::MAX, 1, "add overflow"),
        Err("add overflow".to_owned())
    );
}
