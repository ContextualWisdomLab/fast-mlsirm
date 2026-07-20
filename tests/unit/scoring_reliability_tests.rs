use super::*;

#[test]
fn empirical_reliability_tracks_signal_to_noise() {
    // wide score spread + small SEs -> high rho; flat scores -> low rho
    let n = 200usize;
    let eap: Vec<f64> = (0..n).map(|p| -2.0 + 4.0 * p as f64 / n as f64).collect();
    let sd_small = vec![0.3_f64; n];
    let sd_large = vec![1.5_f64; n];
    let hi = empirical_reliability(&eap, &sd_small, n, 1).unwrap()[0];
    let lo = empirical_reliability(&eap, &sd_large, n, 1).unwrap()[0];
    assert!(hi > 0.85, "high-information scale must be reliable: {hi}");
    assert!(
        lo < hi - 0.2,
        "noisier scale must be less reliable: {lo} vs {hi}"
    );
    assert!(empirical_reliability(&eap, &sd_small, 3, 1).is_err());
    assert!(empirical_reliability(&[], &[], 2, 0).is_err());
    assert!(empirical_reliability(&[0.0, f64::NAN], &[0.3, 0.3], 2, 1).is_err());
    assert!(empirical_reliability(&[0.0, 1.0], &[-0.3, 0.3], 2, 1).is_err());
    assert!(empirical_reliability(&[0.0, 1.0], &[0.3, f64::INFINITY], 2, 1).is_err());
}
