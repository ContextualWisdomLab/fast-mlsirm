//! Exact-rational midpoint normal quantiles for caller-selected node counts.
//!
//! Numerical basis: Wichura (1988, pp. 478–480, 482–484) for the AS 241 seed
//! and direct-tail evaluation; NIST DLMF §7.9, Eq. 7.9.1 for scaled erfc.
//! Twofold residual correction is a repository-derived extension, not a
//! correctly-rounded guarantee from either source. Independent fixed-reference
//! tests retain a one-ULP criterion; no universal error bound is asserted.
//!
//! References
//! ----------
//! Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the normal
//! distribution. *Applied Statistics, 37*(3), 477–484.
//! https://doi.org/10.2307/2347330
//!
//! National Institute of Standards and Technology. (n.d.). Continued fractions
//! (Eq. 7.9.1). *Digital Library of Mathematical Functions*.
//! https://dlmf.nist.gov/7.9
#[derive(Clone, Copy, Debug)]
struct DD {
    hi: f64,
    lo: f64,
}
impl DD {
    const fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }
    fn from(x: f64) -> Self {
        Self::new(x, 0.0)
    }
    fn renorm(self) -> Self {
        let s = self.hi + self.lo;
        Self::new(s, self.lo - (s - self.hi))
    }
    fn add(self, b: Self) -> Self {
        let s = self.hi + b.hi;
        let v = s - self.hi;
        Self::new(s, ((b.hi - v) + (self.hi - (s - v))) + self.lo + b.lo).renorm()
    }
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
    fn sub(self, b: Self) -> Self {
        self.add(b.neg())
    }
    fn mul(self, b: Self) -> Self {
        let p = self.hi * b.hi;
        let e = self.hi.mul_add(b.hi, -p) + self.hi * b.lo + self.lo * b.hi + self.lo * b.lo;
        Self::new(p, e).renorm()
    }
    fn div_f64(self, b: f64) -> Self {
        let q = self.hi / b;
        let r = ((-q).mul_add(b, self.hi) + self.lo) / b;
        Self::new(q, r).renorm()
    }
    fn div(self, b: Self) -> Self {
        let q0 = self.hi / b.hi;
        let r0 = self.sub(b.mul(Self::from(q0)));
        let q1 = r0.hi / b.hi;
        let r1 = r0.sub(b.mul(Self::from(q1)));
        Self::from(q0)
            .add(Self::from(q1))
            .add(Self::from(r1.hi / b.hi))
            .renorm()
    }
    fn abs(self) -> Self {
        if self.hi < 0.0 {
            self.neg()
        } else {
            self
        }
    }
    fn to_f64(self) -> f64 {
        self.hi + self.lo
    }
}

const INV_SQRT_2PI: DD = DD::new(0.3989422804014327, -2.49232720227773e-17);

fn as241(p: f64) -> f64 {
    const A: [f64; 8] = [
        3.3871328727963666080e0,
        1.3314166789178437745e2,
        1.9715909503065514427e3,
        1.3731693765509461125e4,
        4.5921953931549871457e4,
        6.7265770927008700853e4,
        3.3430575583588128105e4,
        2.5090809287301226727e3,
    ];
    const B: [f64; 8] = [
        1.0,
        4.2313330701600911252e1,
        6.8718700749205790830e2,
        5.3941960214247511077e3,
        2.1213794301586595867e4,
        3.9307895800092710610e4,
        2.8729085735721942674e4,
        5.2264952788528545610e3,
    ];
    const C: [f64; 8] = [
        1.42343711074968357734e0,
        4.63033784615654529590e0,
        5.76949722146069140550e0,
        3.64784832476320460504e0,
        1.27045825245236838258e0,
        2.41780725177450611770e-1,
        2.27238449892691845833e-2,
        7.74545014278341407640e-4,
    ];
    const D: [f64; 8] = [
        1.0,
        2.05319162663775882187e0,
        1.67638483018380384940e0,
        6.89767334985100004550e-1,
        1.48103976427480074590e-1,
        1.51986665636164571966e-2,
        5.47593808499534494600e-4,
        1.05075007164441684324e-9,
    ];
    const E: [f64; 8] = [
        6.65790464350110377720e0,
        5.46378491116411436990e0,
        1.78482653991729133580e0,
        2.96560571828504891230e-1,
        2.65321895265761230930e-2,
        1.24266094738807843860e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ];
    const F: [f64; 8] = [
        1.0,
        5.99832206555887937690e-1,
        1.36929880922735805310e-1,
        1.48753612908506148525e-2,
        7.86869131145613259100e-4,
        1.84631831751005468180e-5,
        1.42151175831644588870e-7,
        2.04426310333889978564e-15,
    ];
    fn horner(a: [f64; 8], r: f64) -> f64 {
        a.into_iter().rev().fold(0.0, |v, c| v * r + c)
    }
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        return q * horner(A, r) / horner(B, r);
    }
    let r = (-p.ln()).sqrt(); // called ONLY with direct lower-tail p
    if r <= 5.0 {
        -horner(C, r - 1.6) / horner(D, r - 1.6)
    } else {
        -horner(E, r - 5.0) / horner(F, r - 5.0)
    }
}

// Author-derived DD integrated exponential series for the center only.
fn center_integral(x: DD) -> Result<DD, &'static str> {
    if x.hi == 0.0 && x.lo == 0.0 {
        return Ok(DD::from(0.0));
    }
    if !x.hi.is_finite() || x.hi.abs() > 4.0 {
        return Err("center-series regime |z|>4 unsupported");
    }
    let neg_half_x2 = x.mul(x).mul(DD::from(-0.5));
    let mut term = x;
    let mut sum = x;
    for j in 1..512 {
        term = term
            .mul(neg_half_x2)
            .mul(DD::from((2 * j - 1) as f64))
            .div_f64(j as f64)
            .div_f64((2 * j + 1) as f64);
        sum = sum.add(term);
        if !sum.hi.is_finite() || !term.hi.is_finite() {
            return Err("nonfinite series");
        }
        if j > 8 && term.hi.abs() < 1e-34 * x.hi.abs() {
            return Ok(sum.mul(INV_SQRT_2PI));
        }
    }
    Err("series exhausted")
}

fn dd_from_u128(value: u128) -> DD {
    const BASE: u128 = 1u128 << 32;
    let high = (value / BASE) as f64;
    let low = (value % BASE) as f64;
    DD::from(high).mul(DD::from(BASE as f64)).add(DD::from(low))
}

// DLMF 7.9.1 gives sqrt(pi) exp(z^2) erfc(z) as this positive continued
// fraction for Re(z)>0.  The finite-depth result is accepted only after two
// successive doubled depths agree; no tail clipping or arbitrary depth cap is
// used as an accuracy claim.
fn scaled_erfc(z: DD) -> Result<DD, &'static str> {
    if !z.hi.is_finite() || z.hi <= 0.0 {
        return Err("invalid positive tail argument");
    }
    let z2 = z.mul(z);
    let mut previous = DD::from(0.0);
    let mut depth = 8usize;
    for _ in 0..12 {
        let mut inner = DD::from(1.0);
        for level in (1..=depth).rev() {
            let coefficient = DD::from(level as f64 * 0.5);
            inner = if level % 2 == 1 {
                z2.add(coefficient.div(inner))
            } else {
                DD::from(1.0).add(coefficient.div(inner))
            };
        }
        let value = z.div(inner);
        if !value.hi.is_finite() || value.hi <= 0.0 {
            return Err("nonfinite continued fraction");
        }
        if depth > 8 {
            let delta = value.sub(previous).abs();
            if delta.hi <= 2e-32 * value.hi {
                return Ok(value);
            }
        }
        previous = value;
        depth *= 2;
    }
    Err("continued fraction refinement exhausted")
}

fn tail_integral(x: DD) -> Result<(DD, DD), &'static str> {
    if !x.hi.is_finite() || x.hi <= 4.0 {
        return Err("tail-series regime requires x>4");
    }
    let z = x.mul(DD::from(0.707106781186547524400844362104849_f64));
    let mills = scaled_erfc(z)?.div_f64(2.0_f64.sqrt());
    let density = DD::from((-0.5 * x.hi * x.hi).exp()).mul(INV_SQRT_2PI);
    Ok((density.mul(mills), density))
}

fn quantile(n: u64, k: u64) -> Result<f64, &'static str> {
    if n == 0 || k == 0 || k > n {
        return Err("invalid n or k");
    }
    let twice_k = (k as u128) * 2;
    let n128 = n as u128;
    let center = n128 + 1;
    if twice_k == center {
        return Ok(0.0);
    }
    let upper = twice_k > center;
    let num = if upper {
        twice_k - n128 - 1
    } else {
        center - twice_k
    };
    let den = n128 * 2;
    let target = dd_from_u128(num).div(dd_from_u128(den));
    let j = if upper {
        n128 + 1 - k as u128
    } else {
        k as u128
    };
    let tail_target = dd_from_u128(j * 2 - 1).div(dd_from_u128(den));
    let mut x = DD::from(-as241(tail_target.to_f64()));
    if !x.hi.is_finite() {
        return Err("nonfinite seed");
    }
    for _ in 0..12 {
        let (residual, density) = if x.hi.abs() <= 4.0 {
            (
                center_integral(x)?.sub(target),
                DD::from((-0.5 * x.hi * x.hi).exp()).mul(INV_SQRT_2PI),
            )
        } else {
            let (q, density) = tail_integral(x.abs())?;
            (q.sub(tail_target), density)
        };
        if !density.hi.is_finite() || density.hi <= 0.0 {
            return Err("invalid Newton density");
        }
        let step = residual.div(density);
        if !step.hi.is_finite() {
            return Err("nonfinite Newton step");
        }
        x = if x.hi.abs() <= 4.0 {
            x.sub(step)
        } else {
            x.abs().add(step).neg()
        };
        if step.hi.abs() <= 1e-15 * x.hi.abs() {
            let check = if x.hi.abs() <= 4.0 {
                center_integral(x)?.sub(target).abs().div(density)
            } else {
                tail_integral(x.abs())?
                    .0
                    .sub(tail_target)
                    .abs()
                    .div(density)
            };
            if check.hi > 2e-14 * x.hi.abs() {
                return Err("residual did not settle");
            }
            let z = x.abs().to_f64();
            return Ok(if upper { z } else { -z });
        }
    }
    Err("Newton exhausted")
}

/// Compute a single exact-rational midpoint quantile without allocating a grid.
///
/// Basis: Wichura (1988, pp. 478–480, 482–484), NIST DLMF §7.9, Eq. 7.9.1;
/// full references and the repository-derived correction scope are above.
/// Count and position are required, with no package count cap or default.
pub fn rational_midpoint_quantile(n_nodes: u64, position: u64) -> Result<f64, String> {
    quantile(n_nodes, position).map_err(str::to_owned)
}

/// Construct antisymmetric float64 nodes and uniform probability weights.
///
/// Nodes target Phi^-1((2k-1)/(2n)); Wichura (1988, pp. 478–480, 482–484)
/// supplies the seed and DLMF §7.9, Eq. 7.9.1 the direct-tail fraction.
/// No count default/cap is used. Platform capacity/allocation failures reject
/// explicitly and are not a reduced-precision fallback. Full references above.
pub fn equal_probability_normal_nodes(n_nodes: u64) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n_nodes == 0 {
        return Err("n_nodes must be >= 1".into());
    }
    let count = usize::try_from(n_nodes)
        .map_err(|_| "requested node array exceeds platform index range")?;
    let mut nodes = Vec::new();
    let mut weights = Vec::new();
    nodes
        .try_reserve_exact(count)
        .map_err(|_| "requested node array exceeds allocation capacity")?;
    weights
        .try_reserve_exact(count)
        .map_err(|_| "requested weight array exceeds allocation capacity")?;
    nodes.resize(count, 0.0);
    weights.resize(count, 1.0 / n_nodes as f64);
    for i in 0..count / 2 {
        let upper = rational_midpoint_quantile(n_nodes, n_nodes - i as u64)?;
        nodes[i] = -upper;
        nodes[count - 1 - i] = upper;
    }
    Ok((nodes, weights))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ulp_distance(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        if a.is_nan() || b.is_nan() {
            return u64::MAX;
        }
        let ia = a.to_bits() as i64;
        let ib = b.to_bits() as i64;
        let oa = if ia < 0 { i64::MIN - ia } else { ia };
        let ob = if ib < 0 { i64::MIN - ib } else { ib };
        oa.abs_diff(ob)
    }

    const REFS: [(u64, u64, f64); 43] = [
        (1u64, 1u64, 0.00000000000000000e+00),
        (2u64, 1u64, -6.74489750196081705e-01),
        (2u64, 2u64, 6.74489750196081705e-01),
        (5u64, 1u64, -1.28155156554460037e+00),
        (5u64, 2u64, -5.24400512708040778e-01),
        (5u64, 3u64, 0.00000000000000000e+00),
        (5u64, 4u64, 5.24400512708040778e-01),
        (5u64, 5u64, 1.28155156554460037e+00),
        (11u64, 1u64, -1.69062162958489814e+00),
        (11u64, 2u64, -1.09680356209351326e+00),
        (11u64, 3u64, -7.47858594763302187e-01),
        (11u64, 4u64, -4.72789120992267331e-01),
        (11u64, 5u64, -2.29884117579232106e-01),
        (11u64, 6u64, 0.00000000000000000e+00),
        (11u64, 7u64, 2.29884117579232106e-01),
        (11u64, 8u64, 4.72789120992267331e-01),
        (11u64, 9u64, 7.47858594763302187e-01),
        (11u64, 10u64, 1.09680356209351326e+00),
        (11u64, 11u64, 1.69062162958489814e+00),
        (121u64, 62u64, 2.07174180839431142e-02),
        (121u64, 63u64, 4.14437330939965987e-02),
        (121u64, 90u64, 6.42326583935750151e-01),
        (121u64, 108u64, 1.21821969864872082e+00),
        (121u64, 120u64, 2.24460675970431556e+00),
        (121u64, 121u64, 2.64107005682493634e+00),
        (241u64, 122u64, 1.04011347333021597e-02),
        (241u64, 123u64, 2.08033948510320532e-02),
        (241u64, 180u64, 6.58256314557857714e-01),
        (241u64, 216u64, 1.24912801012920083e+00),
        (241u64, 240u64, 2.49917925320797263e+00),
        (241u64, 241u64, 2.86657657755353812e+00),
        (481u64, 242u64, 5.21130898204402206e-03),
        (481u64, 243u64, 1.04227594962738972e-02),
        (481u64, 360u64, 6.66334207920086907e-01),
        (481u64, 432u64, 1.26513983362044025e+00),
        (481u64, 480u64, 2.73505389951621547e+00),
        (481u64, 481u64, 3.07870819465790335e+00),
        (961u64, 482u64, 2.60835704157123122e-03),
        (961u64, 483u64, 5.21673182931952327e-03),
        (961u64, 720u64, 6.70402161980269584e-01),
        (961u64, 864u64, 1.27329429150145734e+00),
        (961u64, 960u64, 2.95548802264266453e+00),
        (961u64, 961u64, 3.27931809089579751e+00),
    ];
    #[test]
    fn accuracy_43_refs() {
        let mut failed = Vec::new();
        let mut max = 0;
        for &(n, k, r) in &REFS {
            let z = quantile(n, k).expect("finite");
            let d = ulp_distance(z, r);
            max = max.max(d);
            println!(
                "ROW n={n} k={k} actual_bits={} reference_bits={} ulp={d}",
                z.to_bits(),
                r.to_bits()
            );
            if d > 1 {
                failed.push((n, k, d));
            }
        }
        println!("MAX_ULP={max} REFERENCE_COUNT={}", REFS.len());
        assert!(failed.is_empty(), "failures={failed:?}");
    }
    #[test]
    fn invariants() {
        assert!(quantile(0, 1).is_err());
        assert!(quantile(5, 0).is_err());
        assert!(quantile(5, 6).is_err());
        assert!(quantile((1u64 << 52) + 1, 1).is_ok());
        assert!(quantile(u64::MAX, 1).is_ok());
        assert_eq!(quantile(1, 1).unwrap().to_bits(), 0);
        assert_eq!(quantile(5, 3).unwrap().to_bits(), 0);
        for n in [1u64, 2, 5, 11, 120, 121, 240, 241, 480, 481, 960, 961] {
            let mut previous = f64::NEG_INFINITY;
            let mut total_weight = DD::from(0.0);
            let weight = DD::from(1.0).div_f64(n as f64);
            for k in 1..=n {
                let z = quantile(n, k).unwrap();
                assert!(z > previous);
                previous = z;
                let opposite = quantile(n, n + 1 - k).unwrap();
                assert_eq!(z, -opposite);
                if 2 * k == n + 1 {
                    assert_eq!(z.to_bits(), 0);
                }
                total_weight = total_weight.add(weight);
            }
            assert!((total_weight.to_f64() - 1.0).abs() <= f64::EPSILON);
            println!(
                "INVARIANT n={n} count={n} weights_sum={}",
                total_weight.to_f64()
            );
        }
    }
    #[test]
    fn unsupported_tail_guard() {
        assert!(quantile(1000000000, 1).is_ok());
    }
    // The first practical uncapped-node boundary must not be rejected by the spike.
    #[test]
    fn uncapped_count_tail_requires_supported_refinement() {
        let value = quantile(2_000_001, 2_000_001);
        assert!(value.is_ok(), "large-count tail not supported: {value:?}");
    }

    #[test]
    fn independent_scalar_tail_accuracy() {
        let refs = [
            (
                2000001u64,
                2000001u64,
                f64::from_bits(4617345143689277159u64),
            ),
            (1000000000u64, 1u64, f64::from_bits(13841936639662689543u64)),
            (
                4503599627370497u64,
                1u64,
                f64::from_bits(13844183212903688786u64),
            ),
            (
                18446744073709551615u64,
                1u64,
                f64::from_bits(13844715627112426260u64),
            ),
            (
                4503599627370497u64,
                2251799813685250u64,
                f64::from_bits(4378639663258543876u64),
            ),
        ];
        let mut failures = Vec::new();
        for (n, k, reference) in refs {
            let actual = quantile(n, k).expect("supported scalar");
            let distance = ulp_distance(actual, reference);
            println!(
                "SCALAR n={n} k={k} bits={} ref={} ulp={distance}",
                actual.to_bits(),
                reference.to_bits()
            );
            if distance > 1 {
                failures.push((n, k, distance));
            }
        }
        assert!(
            failures.is_empty(),
            "scalar accuracy failures: {failures:?}"
        );
    }

    #[test]
    fn large_count_near_center_is_supported() {
        for (n, k) in [(u64::MAX, (1u64 << 63) + 1), (1u64 << 63, 1u64 << 62)] {
            let z = quantile(n, k);
            assert!(z.is_ok(), "near-center count n={n}, k={k}: {z:?}");
        }
    }

    #[test]
    fn generated_nodes_use_requested_counts() {
        for n in [1u64, 2, 5, 11, 121, 241, 481, 961] {
            let (nodes, weights) = equal_probability_normal_nodes(n).unwrap();
            assert_eq!(nodes.len(), n as usize);
            assert_eq!(weights.len(), n as usize);
            for i in 0..nodes.len() {
                assert_eq!(nodes[i], -nodes[nodes.len() - 1 - i]);
                assert!(nodes[i].is_finite());
                assert_eq!(weights[i], 1.0 / n as f64);
            }
            if n % 2 == 1 {
                assert_eq!(nodes[nodes.len() / 2].to_bits(), 0);
            }
        }
        assert!(equal_probability_normal_nodes(0).is_err());
        assert!(equal_probability_normal_nodes(u64::MAX).is_err());
    }
}
