//! Deterministic person-axis chunking for CPU E-step reductions (#2002).
//!
//! Floating-point addition is not associative (Higham, 2002, §§4.1–4.2), so a
//! reduction whose association tree depends on the runtime thread count is
//! not reproducible. This module fixes the association tree by:
//!
//! 1. partitioning the person index set into a caller-chosen number of chunks
//!    whose boundaries depend only on `n_persons` and `n_chunks` (never on
//!    the thread count);
//! 2. computing each chunk's partial sums independently (persons within a
//!    chunk stay in ascending index order);
//! 3. folding chunk partials in **chunk-index order**.
//!
//! Work is scheduled on a **local** `rayon::ThreadPool` sized to the caller's
//! `n_threads` (Rayon `ThreadPoolBuilder`; never `build_global`), so one fit
//! cannot steal cores from a concurrent outer bootstrap (#2001).
//!
//! Person contributions to the Bock–Aitkin E-step are independent (Bock &
//! Aitkin, 1981; Gibbons et al., 2007, reduced bifactor MML), so reordering
//! across chunks changes only the floating-point association, not the
//! mathematical sum.
//!
//! # References (APA 7th ed.)
//!
//! Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
//! of item parameters: Application of an EM algorithm. *Psychometrika,
//! 46*(4), 443–459. https://doi.org/10.1007/BF02293801
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
//! Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover,
//! A. (2007). Full-information item bifactor analysis of graded response
//! data. *Applied Psychological Measurement, 31*(1), 4–19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Higham, N. J. (2002). *Accuracy and stability of numerical algorithms*
//! (2nd ed.). SIAM. https://doi.org/10.1137/1.9780898718027 (floating-point
//! summation non-associativity, §§4.1–4.2)

use rayon::prelude::*;

/// Half-open person range `[start, end)` for chunk `chunk_idx` of `n_chunks`.
///
/// Chunk size is `ceil(n_persons / n_chunks)`, independent of any thread
/// count. Trailing chunks may be empty when `n_chunks > n_persons`.
///
/// # Panics
///
/// Panics if `n_chunks == 0` or `chunk_idx >= n_chunks`.
#[inline]
pub(crate) fn person_chunk_range(
    n_persons: usize,
    n_chunks: usize,
    chunk_idx: usize,
) -> (usize, usize) {
    assert!(n_chunks >= 1, "n_chunks must be >= 1");
    assert!(
        chunk_idx < n_chunks,
        "chunk_idx {chunk_idx} out of range for n_chunks {n_chunks}"
    );
    let chunk_size = n_persons.div_ceil(n_chunks);
    let start = chunk_idx.saturating_mul(chunk_size);
    if start >= n_persons {
        return (n_persons, n_persons);
    }
    let end = (start + chunk_size).min(n_persons);
    (start, end)
}

/// Map each person-chunk on a local rayon pool, then fold results in
/// chunk-index order.
///
/// `map_chunk(start, end)` must be pure with respect to shared mutable state
/// (thread-local scratch only). The returned `Vec` is indexed by chunk, so
/// callers reduce with a sequential left fold over `0..n_chunks`.
pub(crate) fn map_person_chunks<T, F>(
    n_persons: usize,
    n_chunks: usize,
    n_threads: usize,
    map_chunk: F,
) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    if n_chunks < 1 {
        return Err("e_step_n_chunks must be >= 1".into());
    }
    if n_threads < 1 {
        return Err("e_step_n_threads must be >= 1".into());
    }
    // Single chunk: keep the historical person-order association (no rayon).
    if n_chunks == 1 {
        let (start, end) = person_chunk_range(n_persons, 1, 0);
        return Ok(vec![map_chunk(start, end)]);
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_threads)
        .build()
        .map_err(|e| format!("failed to build E-step rayon pool: {e}"))?;
    // IndexedParallelIterator::collect preserves chunk-index order
    // (rayon::iter::IndexedParallelIterator), so the Vec is already in
    // reduction order before the caller's ordered fold.
    let partials = pool.install(|| {
        (0..n_chunks)
            .into_par_iter()
            .map(|c| {
                let (start, end) = person_chunk_range(n_persons, n_chunks, c);
                map_chunk(start, end)
            })
            .collect::<Vec<_>>()
    });
    Ok(partials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bounds_depend_only_on_n_persons_and_n_chunks() {
        // 10 persons / 3 chunks → size 4: [0,4), [4,8), [8,10)
        assert_eq!(person_chunk_range(10, 3, 0), (0, 4));
        assert_eq!(person_chunk_range(10, 3, 1), (4, 8));
        assert_eq!(person_chunk_range(10, 3, 2), (8, 10));
    }

    #[test]
    fn trailing_chunks_may_be_empty_when_n_chunks_exceeds_n_persons() {
        assert_eq!(person_chunk_range(3, 5, 0), (0, 1));
        assert_eq!(person_chunk_range(3, 5, 1), (1, 2));
        assert_eq!(person_chunk_range(3, 5, 2), (2, 3));
        assert_eq!(person_chunk_range(3, 5, 3), (3, 3));
        assert_eq!(person_chunk_range(3, 5, 4), (3, 3));
    }

    #[test]
    fn ordered_map_is_bit_identical_across_thread_counts() {
        let n_persons = 17usize;
        let n_chunks = 4usize;
        let map = |start: usize, end: usize| -> f64 {
            // Non-associative-looking sum of many distinct terms.
            (start..end).map(|p| (p as f64 + 1.0).sin()).sum()
        };
        let a = map_person_chunks(n_persons, n_chunks, 1, map).unwrap();
        let b = map_person_chunks(n_persons, n_chunks, 8, map).unwrap();
        assert_eq!(a.len(), n_chunks);
        assert_eq!(b.len(), n_chunks);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "chunk partials must be bit-identical across thread counts"
            );
        }
        let fold = |parts: &[f64]| parts.iter().fold(0.0f64, |acc, x| acc + *x);
        assert_eq!(fold(&a).to_bits(), fold(&b).to_bits());
    }
}
