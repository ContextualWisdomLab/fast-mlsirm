//! EM iteration progress reports for long Bock–Aitkin MML fits.
//!
//! The observed-data marginal log-likelihood is already evaluated each
//! E-step for convergence monitoring; exporting it requires no extra
//! quadrature (Bock & Aitkin, 1981, pp. 445, 447–448).
//!
//! # References (APA 7th ed.)
//!
//! Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
//! of item parameters: Application of an EM algorithm. *Psychometrika,
//! 46*(4), 443–459. https://doi.org/10.1007/BF02293801 (full text read:
//! p. 445 eqs. 5–6 define the marginal log-likelihood; p. 447 E-step
//! recomputes pattern marginals `P_l` each cycle; p. 448 notes the
//! procedure satisfies the marginal likelihood equations)

use std::ops::ControlFlow;

/// One EM E-step's already-computed observed-data marginal log-likelihood.
///
/// `iteration` is the 0-based index of completed E-step evaluations within
/// the current multi-start run (matches `loglik_trace.len() - 1` after the
/// report is emitted). `delta_loglik` is `None` on the first evaluation of
/// a start and `Some(current - previous)` thereafter — the same relative
/// change used for the tolerance check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EmIterationProgress {
    pub iteration: usize,
    pub loglik: f64,
    /// `None` on the first E-step of a start; else `loglik - previous`.
    pub delta_loglik: Option<f64>,
    /// Multi-start index in `0..n_starts`.
    pub start: usize,
}

/// Optional mutable progress sink; `None` keeps the fitter silent.
///
/// Returning [`ControlFlow::Break`] cancels the entire multi-start fit before
/// another E-step or start is evaluated.
pub type EmProgressCallback<'a> = dyn FnMut(EmIterationProgress) -> ControlFlow<()> + 'a;
