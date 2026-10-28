//! Orthogonal normal-ogive GRM reporting transforms.

#[derive(Debug)]
pub struct GrmReport {
    pub loadings: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub communality: f64,
    pub uniqueness: f64,
}

/// Transform one item's slopes and decreasing boundary intercepts.
///
/// Independent unit-variance factors only. `scale=1` gives the normal-ogive
/// transform; dividing logistic parameters by a caller-selected positive scale
/// is an approximation, not equality of logistic and normal response functions.
/// The caller must retain that metric distinction in its report.
///
/// Basis: Muraki and Carlson (1995, pp. 74–76, eqs. 23–25; p. 79, eqs. 38–39),
/// and Chalmers (2012, p. 3, eq. 1). No study-specific scale is defaulted.
///
/// Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for
/// polytomous item responses. *Applied Psychological Measurement, 19*(1), 73–90.
/// https://doi.org/10.1177/014662169501900109
/// Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
/// for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
/// https://doi.org/10.18637/jss.v048.i06
pub fn orthogonal_grm_report(
    slopes: &[f64],
    intercepts: &[f64],
    scale: f64,
) -> Result<GrmReport, String> {
    if slopes.is_empty() || intercepts.is_empty() {
        return Err("slopes and intercepts must be nonempty".into());
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err("scale must be finite and positive".into());
    }
    if slopes.iter().chain(intercepts).any(|x| !x.is_finite()) {
        return Err("parameters must be finite".into());
    }
    if intercepts.windows(2).any(|pair| pair[0] <= pair[1]) {
        return Err("boundary intercepts must be strictly decreasing".into());
    }
    // Scale the norm before squaring: finite extreme slopes must not overflow.
    let magnitude = slopes.iter().fold(scale, |m, x| m.max(x.abs()));
    let slope_norm = slopes.iter().fold(0.0_f64, |n, x| n.hypot(x / magnitude));
    let residual = scale / magnitude;
    let norm = slope_norm.hypot(residual);
    let loadings = slopes.iter().map(|x| (x / magnitude) / norm).collect();
    let thresholds: Vec<f64> = intercepts.iter().map(|x| -(x / norm) / magnitude).collect();
    if thresholds.iter().any(|x| !x.is_finite()) {
        return Err("transformed thresholds are not representable as finite f64".into());
    }
    Ok(GrmReport {
        loadings,
        thresholds,
        communality: (slope_norm / norm).powi(2),
        uniqueness: (residual / norm).powi(2),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_ogive_reduction_and_roundtrip() {
        let r = orthogonal_grm_report(&[3.0, 4.0, 0.0], &[2.0, -1.0], 1.0).unwrap();
        let sigma = r.uniqueness.sqrt();
        for (loading, slope) in r.loadings.iter().zip([3.0, 4.0, 0.0]) {
            assert!((loading / sigma - slope).abs() < 1e-14);
        }
        assert!((r.thresholds[0] / sigma + 2.0).abs() < 1e-14);
        assert!((r.communality + r.uniqueness - 1.0).abs() < 1e-14);
    }

    #[test]
    fn explicit_scale_and_extreme_magnitudes() {
        let a = orthogonal_grm_report(&[0.6, -0.8], &[0.5], 1.0).unwrap();
        let b = orthogonal_grm_report(&[1.2, -1.6], &[1.0], 2.0).unwrap();
        assert_eq!(a.loadings, b.loadings);
        assert_eq!(a.thresholds, b.thresholds);
        for factor in [1e-300, 1e300] {
            let r =
                orthogonal_grm_report(&[3.0 * factor, 4.0 * factor], &[factor], factor).unwrap();
            assert!((r.communality - 25.0 / 26.0).abs() < 1e-14);
        }
        let zero = orthogonal_grm_report(&[0.0], &[2.0], 1.0).unwrap();
        assert_eq!(zero.communality, 0.0);
        assert_eq!(zero.uniqueness, 1.0);
    }

    #[test]
    fn rejects_malformed_or_unrepresentable_parameters() {
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(orthogonal_grm_report(&[1.0], &[0.0], scale).is_err());
        }
        for invalid in [f64::NAN, f64::INFINITY] {
            assert!(orthogonal_grm_report(&[invalid], &[0.0], 1.0).is_err());
            assert!(orthogonal_grm_report(&[1.0], &[invalid], 1.0).is_err());
        }
        assert!(orthogonal_grm_report(&[], &[0.0], 1.0).is_err());
        assert!(orthogonal_grm_report(&[1.0], &[], 1.0).is_err());
        assert!(orthogonal_grm_report(&[1.0], &[0.0, 0.0], 1.0).is_err());
        assert!(orthogonal_grm_report(&[1.0], &[0.0, 1.0], 1.0).is_err());
        assert!(orthogonal_grm_report(&[0.0], &[1e308], 1e-308).is_err());
    }
}
