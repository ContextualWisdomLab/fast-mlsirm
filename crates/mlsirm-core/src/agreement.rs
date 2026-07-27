//! Machine-scoring validation statistics and acceptance gates.
//!
//! Implements the operational criteria of Williamson, Xi & Breyer (2012), "A
//! Framework for Evaluation and Use of Automated Scoring" (Educational
//! Measurement: Issues and Practice 31(1), 2-13) for validating an automated
//! scorer (here: an LLM-as-a-Judge) against human ratings:
//!
//! - quadratic-weighted kappa `κ_w = 1 - Σ w_ij O_ij / Σ w_ij E_ij` with
//!   `w_ij = (i-j)^2/(K-1)^2` (Fleiss & Cohen 1973); gate `κ_w >= .70`
//!   (collapses to Cohen's unweighted kappa for binary labels);
//! - Pearson r on the paired scores; gate `r >= .70`;
//! - degradation vs a human-human baseline `Δ = stat_hh - stat_ah`; gate
//!   `Δ <= .10`;
//! - standardized mean difference `SMD = (M_auto - M_human)/SD_human`; gate
//!   `|SMD| <= .15` overall and `<= .10` within every subgroup;
//! - exact (and adjacent) agreement: reported, explicitly NOT a gate.

/// Cross-tabulate two label vectors with values in `0..k`.
fn joint_counts(a: &[u32], b: &[u32], k: usize) -> Result<Vec<f64>, String> {
    if a.len() != b.len() || a.is_empty() {
        return Err("paired label vectors must be non-empty and equal-length".into());
    }
    let mut table = vec![0.0_f64; k * k];
    for (&x, &y) in a.iter().zip(b) {
        if x as usize >= k || y as usize >= k {
            return Err(format!("labels must be in 0..{k}"));
        }
        table[x as usize * k + y as usize] += 1.0;
    }
    Ok(table)
}

/// Weighted kappa with weights `w_ij = (i-j)^2/(K-1)^2` (quadratic; K >= 2).
/// For `k = 2` this equals Cohen's unweighted kappa.
pub fn quadratic_weighted_kappa(a: &[u32], b: &[u32], k: usize) -> Result<f64, String> {
    if k < 2 {
        return Err("kappa needs at least 2 categories".into());
    }
    let table = joint_counts(a, b, k)?;
    let n = a.len() as f64;
    let mut row = vec![0.0_f64; k];
    let mut col = vec![0.0_f64; k];
    for i in 0..k {
        for j in 0..k {
            row[i] += table[i * k + j];
            col[j] += table[i * k + j];
        }
    }
    let denom_w = ((k - 1) * (k - 1)) as f64;
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for i in 0..k {
        for j in 0..k {
            let w = ((i as f64 - j as f64) * (i as f64 - j as f64)) / denom_w;
            num += w * table[i * k + j] / n;
            den += w * (row[i] / n) * (col[j] / n);
        }
    }
    if den <= 0.0 {
        return Err("degenerate marginals: expected weighted disagreement is zero".into());
    }
    Ok(1.0 - num / den)
}

/// Cohen's unweighted kappa.
pub fn cohen_kappa(a: &[u32], b: &[u32], k: usize) -> Result<f64, String> {
    let table = joint_counts(a, b, k)?;
    let n = a.len() as f64;
    let mut po = 0.0_f64;
    let mut row = vec![0.0_f64; k];
    let mut col = vec![0.0_f64; k];
    for i in 0..k {
        po += table[i * k + i] / n;
        for j in 0..k {
            row[i] += table[i * k + j];
            col[j] += table[i * k + j];
        }
    }
    let pe: f64 = (0..k).map(|i| (row[i] / n) * (col[i] / n)).sum();
    if (1.0 - pe).abs() < 1e-12 {
        return Err("degenerate marginals: chance agreement is 1".into());
    }
    Ok((po - pe) / (1.0 - pe))
}

/// Pearson product-moment correlation of paired scores.
pub fn pearson_r(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.len() != b.len() || a.len() < 2 {
        return Err("paired score vectors must be equal-length with n >= 2".into());
    }
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return Err("zero variance in one of the score vectors".into());
    }
    Ok(sab / (saa.sqrt() * sbb.sqrt()))
}

/// Standardized mean difference, standardized on the HUMAN score SD:
/// `(M_auto - M_human) / SD_human` (Williamson et al. criterion E).
pub fn smd(auto: &[f64], human: &[f64]) -> Result<f64, String> {
    if auto.len() != human.len() || human.len() < 2 {
        return Err("paired score vectors must be equal-length with n >= 2".into());
    }
    let n = human.len() as f64;
    let mh = human.iter().sum::<f64>() / n;
    let ma = auto.iter().sum::<f64>() / n;
    let var_h = human.iter().map(|&v| (v - mh) * (v - mh)).sum::<f64>() / n;
    if var_h <= 0.0 {
        return Err("human scores have zero variance".into());
    }
    Ok((ma - mh) / var_h.sqrt())
}

/// Proportion of exact matches, and matches within +/- 1 category.
pub fn agreement_rates(a: &[u32], b: &[u32]) -> Result<(f64, f64), String> {
    if a.len() != b.len() || a.is_empty() {
        return Err("paired label vectors must be non-empty and equal-length".into());
    }
    let n = a.len() as f64;
    let exact = a.iter().zip(b).filter(|(&x, &y)| x == y).count() as f64 / n;
    let adjacent = a
        .iter()
        .zip(b)
        .filter(|(&x, &y)| (x as i64 - y as i64).abs() <= 1)
        .count() as f64
        / n;
    Ok((exact, adjacent))
}

/// One gate outcome: the statistic, its threshold, and whether it passed.
#[derive(Clone, Debug)]
pub struct Gate {
    pub name: &'static str,
    pub value: f64,
    pub threshold: f64,
    pub pass: bool,
}

/// Conjunctive validation verdict per Williamson et al. (2012).
#[derive(Clone, Debug)]
pub struct ValidationVerdict {
    pub gates: Vec<Gate>,
    /// Reported-only statistics (exact/adjacent agreement).
    pub exact_agreement: f64,
    pub adjacent_agreement: f64,
    pub pass: bool,
}

/// Run the conjunctive acceptance gates on paired (auto, human) labels in
/// `0..k`. `human_human` optionally supplies a double-scored baseline
/// (pairs of human labels) for the degradation criterion; `subgroup` labels
/// each observation for the fairness SMD.
pub fn validate_scoring(
    auto: &[u32],
    human: &[u32],
    k: usize,
    human_human: Option<(&[u32], &[u32])>,
    subgroup: Option<&[u32]>,
) -> Result<ValidationVerdict, String> {
    let auto_f: Vec<f64> = auto.iter().map(|&v| v as f64).collect();
    let human_f: Vec<f64> = human.iter().map(|&v| v as f64).collect();
    let mut gates = Vec::new();

    let qwk = quadratic_weighted_kappa(auto, human, k)?;
    gates.push(Gate {
        name: "qwk",
        value: qwk,
        threshold: 0.70,
        pass: qwk >= 0.70,
    });
    let r = pearson_r(&auto_f, &human_f)?;
    gates.push(Gate {
        name: "pearson_r",
        value: r,
        threshold: 0.70,
        pass: r >= 0.70,
    });
    let s = smd(&auto_f, &human_f)?;
    gates.push(Gate {
        name: "smd",
        value: s,
        threshold: 0.15,
        pass: s.abs() <= 0.15,
    });

    if let Some((h1, h2)) = human_human {
        let hh = quadratic_weighted_kappa(h1, h2, k)?;
        let degradation = hh - qwk;
        gates.push(Gate {
            name: "degradation",
            value: degradation,
            threshold: 0.10,
            pass: degradation <= 0.10,
        });
    }

    if let Some(groups) = subgroup {
        if groups.len() != auto.len() {
            return Err("subgroup labels must match the paired vectors".into());
        }
        let n_groups = groups.iter().map(|&g| g as usize).max().unwrap_or(0) + 1;
        let mut worst: f64 = 0.0;
        for g in 0..n_groups {
            let idx: Vec<usize> = (0..groups.len())
                .filter(|&i| groups[i] as usize == g)
                .collect();
            if idx.len() < 2 {
                continue;
            }
            let ga: Vec<f64> = idx.iter().map(|&i| auto_f[i]).collect();
            let gh: Vec<f64> = idx.iter().map(|&i| human_f[i]).collect();
            let Ok(gs) = smd(&ga, &gh) else {
                continue;
            };
            if gs.abs() > worst.abs() {
                worst = gs;
            }
        }
        gates.push(Gate {
            name: "subgroup_smd",
            value: worst,
            threshold: 0.10,
            pass: worst.abs() <= 0.10,
        });
    }

    let (exact, adjacent) = agreement_rates(auto, human)?;
    let pass = gates.iter().all(|g| g.pass);
    Ok(ValidationVerdict {
        gates,
        exact_agreement: exact,
        adjacent_agreement: adjacent,
        pass,
    })
}

/// Result of Fleiss' multi-rater kappa (`fleiss_kappa`).
///
/// In exact (Conger) mode `z`/`p_value` are NaN and the category vectors
/// are empty, mirroring irr's `kappam.fleiss(exact=TRUE)` which returns
/// neither a test statistic nor category detail.
#[derive(Clone, Debug)]
pub struct FleissKappaResult {
    pub kappa: f64,
    /// Subjects remaining after listwise deletion of rows with missing codes.
    pub subjects_used: usize,
    pub z: f64,
    pub p_value: f64,
    /// Category-wise kappas (classic mode only); NaN for empty categories.
    pub category_kappa: Vec<f64>,
    pub category_z: Vec<f64>,
    pub category_p: Vec<f64>,
}

/// Fleiss' kappa for nominal agreement among `nr` raters over `ns` subjects,
/// with the exact (Conger) chance-agreement variant.
///
/// Reimplements `kappam.fleiss()` from CRAN irr 0.85 (`R/kappam.fleiss.R`,
/// READ in full; algorithm source of truth). The model originates in
/// Fleiss, J. L. (1971), "Measuring nominal scale agreement among many
/// raters," Psychological Bulletin, 76(5), 378-382, and the exact variant in
/// Conger, A. J. (1980), "Integration and generalization of kappas for
/// multiple raters," Psychological Bulletin, 88(2), 322-328 — both cited as
/// origins only (NOT READ); every formula below was verified against the irr
/// R source.
///
/// `ratings` is row-major `ns x nr` with category codes `0..k-1`; a negative
/// code marks a missing rating and drops the whole subject row (listwise, as
/// in R: `ratings[apply(is.na(ratings),1,sum)==0,]`). With `m` used subjects
/// and `ttab[i][j]` = raters assigning subject `i` to category `j`:
///
/// - `agreeP = (1/m) sum_i (sum_j ttab_ij^2 - nr) / (nr(nr-1))`
/// - classic `chanceP = sum_j p_j^2` with `p_j = C_j/(m nr)`, `C_j` column sums
/// - exact `chanceP = sum_j p_j^2 - (1/nr) sum_j s2_j`, `s2_j` the sample
///   variance (divisor `nr-1`) over raters of per-rater category proportions
///   (algebraically equal to R's `sum(apply(rtab,2,var)*(nr-1)/nr)/(nr-1)`)
/// - `kappa = (agreeP - chanceP)/(1 - chanceP)`
///
/// Classic mode adds Fleiss' large-sample test
/// `var = 2[(sum p_j q_j)^2 - sum p_j q_j (q_j - p_j)] /
///        [(sum p_j q_j)^2 m nr (nr-1)]`, `z = kappa/sqrt(var)`,
/// `p = 2(1 - Phi(|z|))`, and category-wise kappas
/// `pjk_j = (sum_i ttab_ij^2 - m nr p_j)/(m nr (nr-1) p_j)`,
/// `kappa_j = (pjk_j - p_j)/(1 - p_j)`, `var_j = 2/(m nr (nr-1))`
/// (computed unconditionally; identical to irr's `detail=TRUE`). Empty
/// categories yield NaN, matching R's 0/0.
///
/// API deviations from R (documented contract, not transcription): codes are
/// index-based `0..k-1` with explicit `k` (R derives factor levels; negative
/// numeric labels must be remapped by the caller since negative = missing
/// here), degenerate `1 - chanceP == 0` is an error (R returns NaN), and the
/// size caps below are safety bounds.
pub fn fleiss_kappa(
    ratings: &[i64],
    ns: usize,
    nr: usize,
    k: usize,
    exact: bool,
) -> Result<FleissKappaResult, String> {
    if ns == 0 {
        return Err("need at least one subject".into());
    }
    if nr < 2 {
        return Err("need at least 2 raters".into());
    }
    if k < 2 {
        return Err("need at least 2 categories".into());
    }
    if ns > 1_000_000 || nr > 10_000 || k > 10_000 {
        return Err("size caps: ns <= 1e6, nr <= 1e4, k <= 1e4".into());
    }
    if ratings.len() != ns * nr {
        return Err(format!(
            "ratings length {} != ns*nr = {}",
            ratings.len(),
            ns * nr
        ));
    }
    for &c in ratings {
        if c >= k as i64 {
            return Err(format!("category code {c} out of range 0..{k}"));
        }
    }
    // Listwise drop of rows containing any negative (missing) code, then
    // classification table ttab[i][j] and per-rater counts. Counts are exact
    // in f64: entries <= nr <= 1e4, sums of squares <= m*nr^2 <= 1e14 < 2^53.
    let mut ttab: Vec<Vec<f64>> = Vec::new();
    let mut rater_counts = vec![vec![0.0_f64; k]; nr];
    for i in 0..ns {
        let row = &ratings[i * nr..(i + 1) * nr];
        if row.iter().any(|&c| c < 0) {
            continue;
        }
        let mut t = vec![0.0_f64; k];
        for (r, &c) in row.iter().enumerate() {
            t[c as usize] += 1.0;
            rater_counts[r][c as usize] += 1.0;
        }
        ttab.push(t);
    }
    let m = ttab.len();
    if m == 0 {
        return Err("all subject rows dropped for missing ratings".into());
    }
    let mf = m as f64;
    let nrf = nr as f64;

    let agree_p: f64 = ttab
        .iter()
        .map(|t| (t.iter().map(|&v| v * v).sum::<f64>() - nrf) / (nrf * (nrf - 1.0)))
        .sum::<f64>()
        / mf;

    let col: Vec<f64> = (0..k)
        .map(|j| ttab.iter().map(|t| t[j]).sum::<f64>())
        .collect();
    let p: Vec<f64> = col.iter().map(|&c| c / (mf * nrf)).collect();
    let mut chance_p: f64 = p.iter().map(|&v| v * v).sum();
    if exact {
        // Sample variance over raters of rtab[r][j] = rater_counts[r][j]/m.
        let mut s2_sum = 0.0_f64;
        for j in 0..k {
            let props: Vec<f64> = (0..nr).map(|r| rater_counts[r][j] / mf).collect();
            let mean = props.iter().sum::<f64>() / nrf;
            let s2 = props.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nrf - 1.0);
            s2_sum += s2;
        }
        chance_p -= s2_sum / nrf;
    }
    let denom = 1.0 - chance_p;
    if denom.abs() < 1e-12 {
        return Err("degenerate marginals: no chance-corrected agreement is defined".into());
    }
    let kappa = (agree_p - chance_p) / denom;

    if exact {
        return Ok(FleissKappaResult {
            kappa,
            subjects_used: m,
            z: f64::NAN,
            p_value: f64::NAN,
            category_kappa: Vec::new(),
            category_z: Vec::new(),
            category_p: Vec::new(),
        });
    }

    let sqrt2 = std::f64::consts::SQRT_2;
    let pq: f64 = p.iter().map(|&v| v * (1.0 - v)).sum();
    let var = 2.0
        * (pq * pq
            - p.iter()
                .map(|&v| v * (1.0 - v) * (1.0 - 2.0 * v))
                .sum::<f64>())
        / (pq * pq * mf * nrf * (nrf - 1.0));
    let z = kappa / var.sqrt();
    let p_value = crate::fitstats::erfc(z.abs() / sqrt2);

    let var_k = 2.0 / (mf * nrf * (nrf - 1.0));
    let mut category_kappa = Vec::with_capacity(k);
    let mut category_z = Vec::with_capacity(k);
    let mut category_p = Vec::with_capacity(k);
    for j in 0..k {
        let sum_sq: f64 = ttab.iter().map(|t| t[j] * t[j]).sum();
        // Empty category: p_j = 0 gives R's 0/0 = NaN, preserved here.
        let pjk = (sum_sq - mf * nrf * p[j]) / (mf * nrf * (nrf - 1.0) * p[j]);
        let kj = (pjk - p[j]) / (1.0 - p[j]);
        let zj = kj / var_k.sqrt();
        category_kappa.push(kj);
        category_z.push(zj);
        category_p.push(if zj.is_finite() {
            crate::fitstats::erfc(zj.abs() / sqrt2)
        } else {
            f64::NAN
        });
    }

    Ok(FleissKappaResult {
        kappa,
        subjects_used: m,
        z,
        p_value,
        category_kappa,
        category_z,
        category_p,
    })
}

/// Result of Light's kappa (`light_kappa`).
#[derive(Clone, Debug)]
pub struct LightKappaResult {
    /// Mean of the pairwise unweighted Cohen's kappas.
    pub value: f64,
    /// Subjects remaining after listwise deletion of rows with missing codes.
    pub subjects_used: usize,
    pub raters: usize,
    /// Pairwise kappas in `(i, j)` order with `i < j`, `i` outer.
    pub kappas: Vec<f64>,
    pub z: f64,
    pub p_value: f64,
}

/// Light's kappa: mean pairwise unweighted Cohen's kappa over `nr` raters,
/// with Light's chance-product z test.
///
/// Reimplements `kappam.light()` from CRAN irr 0.85 (`R/kappam.light.R`,
/// READ in full) together with the unweighted branch of `kappa2()`
/// (`R/kappa2.R`, READ in full); both R sources are the algorithm source of
/// truth. The method originates in Light, R. J. (1971), "Measures of
/// response agreement for qualitative data: Some generalizations and
/// alternatives," Psychological Bulletin, 76(5), 365-377 — cited as origin
/// only (NOT READ); every formula below was verified against the irr R
/// source and an exact-fraction oracle.
///
/// `ratings` is row-major `ns x nr` with integer category codes; a negative
/// code marks a missing rating and drops the whole subject row (listwise, as
/// in R's `na.omit`). With `m` used subjects:
///
/// - each unordered rater pair `(i, j)` yields an unweighted Cohen's kappa
///   `(po - pe)/(1 - pe)` (kappa2.R unweighted branch); R builds each pair's
///   level set from the two selected columns only, but the unweighted value
///   is invariant to unused levels (zero rows/columns change neither the
///   diagonal, the marginals, po, nor pe), so this implementation compacts
///   codes over the full remaining matrix once and reuses [`cohen_kappa`] —
///   an equivalent shortcut, verified in the oracle for every pair of every
///   fixture;
/// - `value` = arithmetic mean of the `C(nr,2)` pairwise kappas;
/// - z test (kappam.light.R lines 31-54): per pair, with category-count
///   vectors `c1, c2` over the full level set,
///   `disrater = sum_{a != b} c1[a] c2[b] = m^2 - sum_a c1[a] c2[a]`;
///   `chanceP = 1 - npairs * prod(disrater / m^2)` (algebraically identical
///   to R's `1 - B/m^(2*npairs)` with `B = npairs * prod(disrater)`, but
///   overflow-safe); `varkappa = chanceP/(m (1 - chanceP))`;
///   `z = value/sqrt(varkappa)`; `p = erfc(|z|/sqrt(2))`.
///
/// API deviations from R (documented contract, not transcription): a pair
/// with `pe == 1` (both raters constant on one shared category) is an error
/// (R yields 0/0 = NaN); `chanceP <= 0` — reachable on valid data, e.g. two
/// identical rows over three raters give `chanceP = -2` — and
/// `1 - chanceP == 0` are errors (R silently produces NaN or infinite
/// variance); all rows dropped is an error; size caps below are safety
/// bounds.
pub fn light_kappa(ratings: &[i64], ns: usize, nr: usize) -> Result<LightKappaResult, String> {
    if ns == 0 {
        return Err("need at least one subject".into());
    }
    if nr < 2 {
        return Err("need at least 2 raters".into());
    }
    if ns > 1_000_000 || nr > 10_000 {
        return Err("size caps: ns <= 1e6, nr <= 1e4".into());
    }
    if ratings.len() != ns * nr {
        return Err(format!(
            "ratings length {} != ns*nr = {}",
            ratings.len(),
            ns * nr
        ));
    }
    for &c in ratings {
        if c > 1i64 << 32 {
            return Err("category codes must be <= 2^32".into());
        }
    }
    // Listwise drop, then compact observed codes to 0..k-1 in sorted order.
    let rows: Vec<&[i64]> = (0..ns)
        .map(|i| &ratings[i * nr..(i + 1) * nr])
        .filter(|row| row.iter().all(|&c| c >= 0))
        .collect();
    let m = rows.len();
    if m == 0 {
        return Err("all subject rows dropped for missing ratings".into());
    }
    let mut levels: Vec<i64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    levels.sort_unstable();
    levels.dedup();
    let k = levels.len();
    if k < 2 {
        return Err("need at least 2 distinct observed categories".into());
    }
    let code = |c: i64| levels.binary_search(&c).unwrap() as u32;
    let cols: Vec<Vec<u32>> = (0..nr)
        .map(|j| rows.iter().map(|r| code(r[j])).collect())
        .collect();

    // Pairwise unweighted Cohen's kappas (mean = Light's value). Counts per
    // level for the z test are accumulated in the same pass.
    let mf = m as f64;
    let counts: Vec<Vec<f64>> = cols
        .iter()
        .map(|col| {
            let mut c = vec![0.0_f64; k];
            for &v in col {
                c[v as usize] += 1.0;
            }
            c
        })
        .collect();
    let mut kappas = Vec::with_capacity(nr * (nr - 1) / 2);
    let mut prod_ratio = 1.0_f64;
    for i in 0..nr - 1 {
        for j in i + 1..nr {
            let kp = cohen_kappa(&cols[i], &cols[j], k)
                .map_err(|e| format!("rater pair ({i},{j}): {e}"))?;
            kappas.push(kp);
            let same: f64 = (0..k).map(|a| counts[i][a] * counts[j][a]).sum();
            // disrater/m^2 with disrater = m^2 - sum_a c1[a]*c2[a].
            prod_ratio *= (mf * mf - same) / (mf * mf);
        }
    }
    let npairs = kappas.len() as f64;
    let value = kappas.iter().sum::<f64>() / npairs;

    let chance_p = 1.0 - npairs * prod_ratio;
    if chance_p <= 0.0 || (1.0 - chance_p).abs() < 1e-12 {
        return Err(
            "degenerate chance product: Light's z statistic is undefined for these marginals"
                .into(),
        );
    }
    let varkappa = chance_p / (mf * (1.0 - chance_p));
    let z = value / varkappa.sqrt();
    let p_value = crate::fitstats::erfc(z.abs() / std::f64::consts::SQRT_2);
    if !z.is_finite() || !p_value.is_finite() {
        return Err("non-finite test statistic".into());
    }

    Ok(LightKappaResult {
        value,
        subjects_used: m,
        raters: nr,
        kappas,
        z,
        p_value,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/agreement_tests.rs"]
mod tests;
