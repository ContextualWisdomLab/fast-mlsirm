//! Robitzsch (2024), Information 15(11), 668, Eq. 2, §3.1 and Table A1.
//! https://doi.org/10.3390/info15110668 (CC BY 4.0).
//! The printed NORM bank is deliberately not standardized or truth-initialized.
pub const A: [f64; 10] = [0.85, 2.01, 1.69, 4.05, 4.77, 2.38, 1.56, 1.61, 1.27, 2.20];
pub const B: [f64; 10] = [
    -1.55, -1.76, -1.22, -1.01, -1.12, -0.89, -0.79, -0.31, -0.32, -0.32,
];

pub fn probability(item: usize, theta: f64) -> f64 {
    1.0 / (1.0 + (-A[item] * (theta - B[item])).exp())
}

/// Deterministic independent response generation for the published NORM DGM.
/// Robitzsch (2024), §3.1: standard normal ability, N=2000/I=10 is one cell.
/// Seeds and the SplitMix64/Box-Muller generator are library-test choices.
pub fn simulate(n: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut uniform = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };
    let mut y = Vec::with_capacity(n * 10);
    for _ in 0..n {
        let theta = (-2.0 * uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * uniform()).cos();
        for item in 0..10 {
            y.push(usize::from(uniform() < probability(item, theta)));
        }
    }
    y
}

/// Within-run raw-scale error summary, not a Monte Carlo bias estimator.
#[allow(dead_code)]
pub fn errors(values: &[f64]) -> (f64, f64) {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    (
        values.iter().sum::<f64>() / values.len() as f64,
        (values.iter().map(|x| x * x).sum::<f64>() / values.len() as f64).sqrt(),
    )
}
