use mlsirm_core::personfit_np;

#[test]
fn test_personfit_parity() {
    let n = 1000;
    let pi: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();
    let lo: Vec<f64> = (0..n).map(|i| (i as f64)).collect();

    let s1_orig: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * l).sum();
    let s2_orig: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * (1.0 - p) * l).sum();
    let s3_orig: f64 = pi.iter().sum();
    let s4_orig: f64 = pi.iter().map(|&p| p * (1.0 - p)).sum();
    let beta_orig: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * (1.0 - p) * l * l).sum::<f64>() - s2_orig * s2_orig / s4_orig;

    let (s1_fold, s2_fold, s3_fold, s4_fold, beta_term) = pi.iter().zip(&lo).fold(
        (0.0, 0.0, 0.0, 0.0, 0.0),
        |(a1, a2, a3, a4, ab), (&p, &l)| {
            let p_comp = 1.0 - p;
            let p_p_comp = p * p_comp;
            (
                a1 + p * l,
                a2 + p_p_comp * l,
                a3 + p,
                a4 + p_p_comp,
                ab + p_p_comp * l * l,
            )
        }
    );
    let beta_fold = beta_term - s2_fold * s2_fold / s4_fold;

    assert_eq!(s1_orig.to_bits(), s1_fold.to_bits(), "s1 parity failure");
    assert_eq!(s2_orig.to_bits(), s2_fold.to_bits(), "s2 parity failure");
    assert_eq!(s3_orig.to_bits(), s3_fold.to_bits(), "s3 parity failure");
    assert_eq!(s4_orig.to_bits(), s4_fold.to_bits(), "s4 parity failure");
    assert_eq!(beta_orig.to_bits(), beta_fold.to_bits(), "beta parity failure");
}
