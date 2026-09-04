//! Nonparametric person-fit statistics for dichotomous responses.
//!
//! # Citation governance
//!
//! `person_fit_np` is a computational port of seven nonparametric
//! person-fit statistics as IMPLEMENTED by the CRAN PerFit R package
//! (READ: `R/G.R`, `R/Gnormed.R`, `R/NCI.R`, `R/U3.R`, `R/ZU3.R`,
//! `R/C.Sato.R`, `R/Cstar.R`, `R/Accessory.R` (`final.PFS`),
//! `R/SanityChecks.R` (`Sanity.prv`) at cran/PerFit commit
//! `c9df433cba3d7b03d16284e832d55785cb90464c`). NOT READ (cited only as
//! referenced by the PerFit source comments): van der Flier (1977, 1980,
//! 1982); Meijer (1994); Tatsuoka & Tatsuoka (1982, 1983); Sato (1975);
//! Harnisch & Linn (1981); the package paper Tendeiro, Meijer & Niessen
//! (2016), Journal of Statistical Software, 74(5).
//!
//! REDUCED SCOPE (adversarial spec review,
//! files/perfit_spec_review.md): complete 0/1 data only. PerFit's
//! missing-value imputation (`MissingValues.R`) and polytomous variants
//! are out of scope; any non-{0,1} entry (including NaN) is an `Err`.
//! Output NaN, by contrast, is a valid statistic value: U3/ZU3/C/C* are
//! NaN for perfect (all-0s/all-1s) rows (R removes them via `Sanity.prv`
//! and `final.PFS` reinserts NA), and degenerate arithmetic (all item
//! proportions equal, zero covariance denominators) yields NaN rather
//! than an error, with any intermediate `Inf` normalized to NaN.
//!
//! Perfect-row values, statistic by statistic (source-faithful):
//! G = 0 (no 0-1 pairs), Gnormed = 0 (`Gnormed.R` sets 0/0 -> 0),
//! NCI = 0 (`NCI.R` applies `1 - 2*res` BEFORE the NaN -> 0 replacement,
//! so perfect rows end at 0, not 1), U3/ZU3/C/C* = NaN.
//!
//! The identity `NCI = 1 - 2*Gnormed` (stated in the `NCI.R`/`Gnormed.R`
//! headers) therefore holds for NON-perfect rows only.
//!
//! Column ordering: R `order(pi, decreasing = TRUE)` is a stable sort by
//! item proportion-correct descending with ties broken by ascending
//! original column index; this port reproduces that tie-break exactly
//! (G/Gnormed/NCI/C depend on it when proportions tie).

/// Result of [`person_fit_np`]: one entry per person for each statistic.
#[derive(Debug, Clone)]
pub struct PersonFitNp {
    /// Guttman error count G (`G.R`): number of (0, 1) pairs where the 1
    /// falls on a HARDER item than the 0 in the proportion-correct
    /// descending column order.
    pub g: Vec<f64>,
    /// Normed Guttman errors (`Gnormed.R`): `G / (NC * (I - NC))`; 0 for
    /// perfect rows.
    pub gnormed: Vec<f64>,
    /// Norm conformity index (`NCI.R`): `1 - 2 * Gnormed` for non-perfect
    /// rows; 0 for perfect rows (transform applied before the NaN
    /// replacement in the R source).
    pub nci: Vec<f64>,
    /// U3 (`U3.R`): log-odds Guttman-ness in [0, 1] for non-degenerate
    /// data; NaN for perfect rows.
    pub u3: Vec<f64>,
    /// Standardized U3 (`ZU3.R`); NaN for perfect rows.
    pub zu3: Vec<f64>,
    /// Sato caution index C (`C.Sato.R`): `1 - cov(x_ord, pi_ord) /
    /// cov(guttman, pi_ord)`; NaN for perfect rows.
    pub c_sato: Vec<f64>,
    /// Modified caution index C* (`Cstar.R`), in [0, 1] for
    /// non-degenerate data; NaN for perfect rows.
    pub cstar: Vec<f64>,
}

/// Stable descending order of `pi` with ascending-index tie-break,
/// matching R `order(pi, decreasing = TRUE)`.
fn order_desc_stable(pi: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..pi.len()).collect();
    idx.sort_by(|&a, &b| pi[b].partial_cmp(&pi[a]).unwrap().then(a.cmp(&b)));
    idx
}

/// Sample covariance of `a` and `b` (n-1 denominator, as R `cov`); the
/// denominator cancels in the C ratio but is kept for fidelity.
fn cov1(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (n - 1.0)
}

/// Compute the seven PerFit nonparametric person-fit statistics for a
/// complete dichotomous response matrix (rows = persons, columns =
/// items). See the module docs for the exact contract and provenance.
pub fn person_fit_np(x: &[Vec<f64>]) -> Result<PersonFitNp, String> {
    let n = x.len();
    if n < 1 {
        return Err("person_fit_np: need at least 1 person".to_string());
    }
    let ni = x[0].len();
    if ni < 2 {
        return Err("person_fit_np: need at least 2 items".to_string());
    }
    for (p, row) in x.iter().enumerate() {
        if row.len() != ni {
            return Err(format!(
                "person_fit_np: row {} has {} items, expected {}",
                p,
                row.len(),
                ni
            ));
        }
        for (i, &v) in row.iter().enumerate() {
            if v != 0.0 && v != 1.0 {
                return Err(format!(
                    "person_fit_np: entry ({}, {}) is {}; responses must be exactly 0 or 1 \
                     (missing data is out of scope)",
                    p, i, v
                ));
            }
        }
    }

    let nf = n as f64;
    let nc: Vec<f64> = x.iter().map(|row| row.iter().sum()).collect();
    let pi: Vec<f64> = (0..ni)
        .map(|i| x.iter().map(|row| row[i]).sum::<f64>() / nf)
        .collect();
    let ord = order_desc_stable(&pi);

    // G / Gnormed / NCI (G.R:20-31, Gnormed.R:33-36, NCI.R:33-37).
    let mut g = vec![0.0; n];
    let mut gnormed = vec![0.0; n];
    let mut nci = vec![0.0; n];
    for p in 0..n {
        let row_ord: Vec<f64> = ord.iter().map(|&i| x[p][i]).collect();
        let mut count = 0u64;
        for i0 in 0..ni {
            if row_ord[i0] != 0.0 {
                continue;
            }
            for i1 in (i0 + 1)..ni {
                if row_ord[i1] == 1.0 {
                    count += 1;
                }
            }
        }
        g[p] = count as f64;
        let den = nc[p] * (ni as f64 - nc[p]);
        if den > 0.0 {
            gnormed[p] = g[p] / den;
            nci[p] = 1.0 - 2.0 * gnormed[p];
        } else {
            // Perfect row: Gnormed.R and NCI.R both replace NaN by 0,
            // NCI.R AFTER the 1-2*res transform -> both end at 0.
            gnormed[p] = 0.0;
            nci[p] = 0.0;
        }
    }

    let perfect: Vec<bool> = nc.iter().map(|&c| c == 0.0 || c == ni as f64).collect();

    // Log-odds with infinities zeroed (U3.R:30-31).
    let lo: Vec<f64> = pi
        .iter()
        .map(|&p| {
            let v = (p / (1.0 - p)).ln();
            if v.is_finite() {
                v
            } else {
                0.0
            }
        })
        .collect();
    let mut lo_desc = lo.clone();
    lo_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut lo_asc = lo.clone();
    lo_asc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cum = |v: &[f64]| -> Vec<f64> {
        let mut acc = 0.0;
        v.iter()
            .map(|&e| {
                acc += e;
                acc
            })
            .collect()
    };
    let cum_desc = cum(&lo_desc);
    let cum_asc = cum(&lo_asc);

    // ZU3 complete-data scalars (ZU3.R:34-38; for complete data each
    // `rowSums(pos.no.NAs %*% v)` term is the scalar sum(v) per row —
    // verified in the adversarial spec review).
    let s1: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * l).sum();
    let s2: f64 = pi.iter().zip(&lo).map(|(&p, &l)| p * (1.0 - p) * l).sum();
    let s3: f64 = pi.iter().sum();
    let s4: f64 = pi.iter().map(|&p| p * (1.0 - p)).sum();
    let beta: f64 = pi
        .iter()
        .zip(&lo)
        .map(|(&p, &l)| p * (1.0 - p) * l * l)
        .sum::<f64>()
        - s2 * s2 / s4;

    // Cstar cumulative pi sums (Cstar.R:26-38).
    let mut pi_desc = pi.clone();
    pi_desc.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut pi_asc = pi.clone();
    pi_asc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cum_pi_desc = cum(&pi_desc);
    let cum_pi_asc = cum(&pi_asc);

    let pio: Vec<f64> = ord.iter().map(|&i| pi[i]).collect();

    let nan_if_bad = |v: f64| if v.is_finite() { v } else { f64::NAN };

    let mut u3 = vec![f64::NAN; n];
    let mut zu3 = vec![f64::NAN; n];
    let mut c_sato = vec![f64::NAN; n];
    let mut cstar = vec![f64::NAN; n];
    for p in 0..n {
        if perfect[p] {
            continue;
        }
        let k = nc[p] as usize; // 1 <= k <= ni-1 here
        let sfirst = cum_desc[k - 1];
        let slast = cum_asc[k - 1];
        let xdot_lo: f64 = x[p].iter().zip(&lo).map(|(&v, &l)| v * l).sum();
        let u = (sfirst - xdot_lo) / (sfirst - slast);
        u3[p] = nan_if_bad(u);

        let alpha = s1 + s2 * (nc[p] - s3) / s4;
        let expv = (sfirst - alpha) / (sfirst - slast);
        let varv = beta / ((sfirst - slast) * (sfirst - slast));
        zu3[p] = nan_if_bad((u - expv) / varv.sqrt());

        let row_ord: Vec<f64> = ord.iter().map(|&i| x[p][i]).collect();
        let mut easiest = vec![0.0; ni];
        for e in easiest.iter_mut().take(k) {
            *e = 1.0;
        }
        let den = cov1(&easiest, &pio);
        c_sato[p] = if den != 0.0 {
            nan_if_bad(1.0 - cov1(&row_ord, &pio) / den)
        } else {
            f64::NAN
        };

        let sfp = cum_pi_desc[k - 1];
        let slp = cum_pi_asc[k - 1];
        let xdot_pi: f64 = x[p].iter().zip(&pi).map(|(&v, &q)| v * q).sum();
        cstar[p] = nan_if_bad((sfp - xdot_pi) / (sfp - slp));
    }

    Ok(PersonFitNp {
        g,
        gnormed,
        nci,
        u3,
        zu3,
        c_sato,
        cstar,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/personfit_np_tests.rs"]
mod personfit_np_tests;
