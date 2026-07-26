//! Standard-setting methods.
//!
//! # Citation governance
//!
//! `hofstee` is a computational port of the Hofstee compromise
//! standard-setting method as IMPLEMENTED by the psychometricsGP R
//! package's `fn_plot_hofstee()` (READ: `R/fn_plot_hofstee.R`, author
//! Martin Roberts — the ONLY inspectable implementation found; a GitHub
//! code search surfaced no second one, so this is a single-source port,
//! stated openly). NOT READ: Hofstee, W. K. B. (1983), "The case for
//! compromise in educational selection and grading" — the method is cited
//! only as implemented by psychometricsGP. Plotting, global-environment
//! side effects, and console output of the R function are out of scope.
//!
//! The R function builds a piecewise-linear cumulative relative frequency
//! curve (ogive) over integer score bins 0..=100 (R `cut(x,
//! breaks=-1:100, include.lowest=TRUE)`: right-closed bins `(s-1, s]`,
//! first bin `[-1, 0]`, so a fractional score rounds UP to the next
//! integer bin) and intersects it with the downward Hofstee diagonal from
//! `(min_cut, max_fail)` to `(max_cut, min_fail)`. If they do not cross,
//! the R fallback sets the cut to `min_cut` when the strict fail rate
//! `100·#{v < min_cut}/n` exceeds `max_fail` (reported fail rate rounded
//! UP to two decimals, `ceil(fr·100)/100`), and to `max_cut` otherwise
//! (fail rate `100·#{v < max_cut}/n` rounded DOWN, `floor(fr·100)/100`) —
//! two-decimal DIRECTED rounding, ported verbatim. Note the R source's
//! internal inconsistency, ported faithfully: the ogive counts `<= s`
//! while the fallback counts `< cut`.
//!
//! REDUCED SCOPE relative to R (adversarial spec review,
//! files/hofstee_spec_review.md): R-faithful behavior is claimed only for
//! non-collinear, non-zero-length diagonal configurations. R finds the
//! crossing via `spatstat::crossing.psp`, whose degenerate semantics were
//! NOT verified (no R runtime available), so: a collinear OVERLAP between
//! an ogive segment and the diagonal is an error here, a zero-length
//! diagonal (`min_cut == max_cut && min_fail == max_fail`) is an error,
//! and the ascending first-crossing scan is a port tie-break, not proven
//! `crossing.psp` ordering (the crossing is unique whenever it is
//! transversal, because the ogive is nondecreasing and the diagonal is
//! nonincreasing). R `na.omit` drops missing scores; this port REJECTS
//! non-finite scores instead (stricter, documented).

/// Result of the Hofstee compromise standard-setting method.
#[derive(Debug, Clone)]
pub struct HofsteeResult {
    /// Cut score (x of the Hofstee point), in percentage units.
    pub cut_score: f64,
    /// Failure rate (y of the Hofstee point), in percent.
    pub fail_rate: f64,
    /// True when the ogive and the diagonal do not cross and the R
    /// fallback produced the point.
    pub failed: bool,
    /// Cumulative relative frequency in percent at integer scores
    /// `s = 0..=100`: `(#{v <= s} / n) * 100` (R arithmetic order:
    /// divide first, then multiply).
    pub cum_freq_percent: Vec<f64>,
}

/// 2-D cross product `a.0*b.1 - a.1*b.0`.
fn cross(a: (f64, f64), b: (f64, f64)) -> f64 {
    a.0 * b.1 - a.1 * b.0
}

/// Hofstee compromise cut score; see the module header for the contract,
/// source governance, and reduced scope. `scores` are percentages in
/// `[0, 100]`; `min_cut <= max_cut` and `min_fail <= max_fail` bound the
/// acceptable cut score and failure rate (equality allowed, except both
/// at once — a zero-length diagonal is rejected).
pub fn hofstee(
    scores: &[f64],
    min_cut: f64,
    max_cut: f64,
    min_fail: f64,
    max_fail: f64,
) -> Result<HofsteeResult, String> {
    if scores.is_empty() {
        return Err("scores must be non-empty".to_string());
    }
    for &v in scores {
        if !v.is_finite() {
            return Err("scores must be finite".to_string());
        }
        if !(0.0..=100.0).contains(&v) {
            return Err("scores must lie in [0, 100]".to_string());
        }
    }
    for (name, p) in [
        ("min_cut", min_cut),
        ("max_cut", max_cut),
        ("min_fail", min_fail),
        ("max_fail", max_fail),
    ] {
        if !p.is_finite() || !(0.0..=100.0).contains(&p) {
            return Err(format!("{name} must be finite and in [0, 100]"));
        }
    }
    if min_cut > max_cut {
        return Err("min_cut must not exceed max_cut".to_string());
    }
    if min_fail > max_fail {
        return Err("min_fail must not exceed max_fail".to_string());
    }
    if min_cut == max_cut && min_fail == max_fail {
        return Err(
            "zero-length Hofstee diagonal (min_cut == max_cut and min_fail == max_fail) \
             is unsupported"
                .to_string(),
        );
    }

    // Binning: bin s counts scores in (s-1, s]; v = 0 falls in [-1, 0].
    let n = scores.len() as f64;
    let mut freq = [0usize; 101];
    for &v in scores {
        let s = if v > 0.0 { v.ceil() as usize } else { 0 };
        freq[s] += 1;
    }
    let mut cum_freq_percent = Vec::with_capacity(101);
    let mut acc = 0usize;
    for f in freq {
        acc += f;
        cum_freq_percent.push((acc as f64 / n) * 100.0);
    }

    // Diagonal from (min_cut, max_fail) down to (max_cut, min_fail).
    let q1 = (min_cut, max_fail);
    let d2 = (max_cut - min_cut, min_fail - max_fail);

    for s in 0..100 {
        let p1 = (s as f64, cum_freq_percent[s]);
        let d1 = (1.0, cum_freq_percent[s + 1] - cum_freq_percent[s]);
        let r = (q1.0 - p1.0, q1.1 - p1.1);
        let denom = cross(d1, d2);
        if denom == 0.0 {
            // Parallel. Collinear overlap is out of scope (crossing.psp
            // semantics unverified); plain parallel non-overlap is skipped.
            if cross(r, d1) == 0.0 {
                // Collinear: overlapping x-ranges? The ogive segment spans
                // [s, s+1]; the diagonal spans [min_cut, max_cut] (or a
                // single x when vertical — vertical can't be parallel to
                // d1 whose x-extent is 1, so x-interval logic suffices).
                let lo = min_cut.max(s as f64);
                let hi = max_cut.min(s as f64 + 1.0);
                if lo <= hi {
                    return Err(
                        "collinear overlap between score ogive and Hofstee diagonal \
                         is unsupported"
                            .to_string(),
                    );
                }
            }
            continue;
        }
        let t = cross(r, d2) / denom;
        let u = cross(r, d1) / denom;
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            return Ok(HofsteeResult {
                cut_score: p1.0 + t * d1.0,
                fail_rate: p1.1 + t * d1.1,
                failed: false,
                cum_freq_percent,
            });
        }
    }

    // R fallback: strict '<' counts, two-decimal directed rounding.
    let fr1 = (scores.iter().filter(|&&v| v < min_cut).count() as f64 / n) * 100.0;
    if fr1 > max_fail {
        return Ok(HofsteeResult {
            cut_score: min_cut,
            fail_rate: (fr1 * 100.0).ceil() / 100.0,
            failed: true,
            cum_freq_percent,
        });
    }
    let fr2 = (scores.iter().filter(|&&v| v < max_cut).count() as f64 / n) * 100.0;
    Ok(HofsteeResult {
        cut_score: max_cut,
        fail_rate: (fr2 * 100.0).floor() / 100.0,
        failed: true,
        cum_freq_percent,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/standard_setting_tests.rs"]
mod tests;
