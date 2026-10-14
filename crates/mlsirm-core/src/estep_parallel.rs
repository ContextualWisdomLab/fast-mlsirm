//! Deterministic person-axis chunking for CPU E-step reductions (#2002).
//!
//! Floating-point addition is not associative (Higham, 2002, §§4.1–4.2), so a
//! reduction whose association tree depends on the runtime thread count is
//! not reproducible. This module fixes the association tree by:
//!
//! 1. partitioning the person index set into a caller-chosen number of chunks
//!    whose boundaries depend only on `n_persons` and `n_chunks` (never on
//!    the thread count);
//! 2. computing each non-empty chunk's partial sums independently (persons
//!    within a chunk stay in ascending index order);
//! 3. folding those partials in **chunk-index order**, at most `n_threads`
//!    partials in flight.
//!
//! The caller owns one local `rayon::ThreadPool` per fit start
//! ([`PersonChunkPool`]; never `build_global`). Repeated E-steps inside that
//! start reuse it. `n_chunks == 1` and `n_threads == 1` stay on the calling
//! thread, so a serial fit does not spawn workers (#2001).
//!
//! Person contributions to the Bock–Aitkin E-step are independent (Bock &
//! Aitkin, 1981; Gibbons et al., 2007, reduced bifactor MML), so reordering
//! across chunks changes only the floating-point association, not the
//! mathematical sum. Trailing empty chunks are omitted: adding an exact zero
//! partial would not change a finite sum (IEEE 754), and it would allocate a
//! full counts tensor per empty chunk.
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

use std::sync::OnceLock;

use rayon::prelude::*;

#[cfg(test)]
use std::cell::Cell;

// Pool builds observed on this thread. Fit code builds the pool on the
// calling thread, so a test can reset the cell, run one start, and read it
// back without crosstalk from cargo's parallel test threads.
#[cfg(test)]
thread_local! {
    pub(crate) static POOL_BUILDS_THIS_THREAD: Cell<usize> = Cell::new(0);
}

#[cfg(test)]
pub(crate) fn reset_pool_builds_this_thread() {
    POOL_BUILDS_THIS_THREAD.with(|builds| builds.set(0));
}

#[cfg(test)]
pub(crate) fn pool_builds_this_thread() -> usize {
    POOL_BUILDS_THIS_THREAD.with(|builds| builds.get())
}

/// Local rayon pool owned by one fit start.
///
/// Construction does not spawn threads. The first parallel window
/// (`n_threads > 1` and at least two non-empty chunks) builds the pool;
/// later E-steps on the same value reuse it. Dropping the value joins the
/// workers, so the pool does not outlive the start.
pub(crate) struct PersonChunkPool {
    n_threads: usize,
    pool: OnceLock<rayon::ThreadPool>,
}

impl PersonChunkPool {
    pub(crate) fn new(n_threads: usize) -> Result<Self, String> {
        if n_threads < 1 {
            return Err("e_step_n_threads must be >= 1".into());
        }
        Ok(Self {
            n_threads,
            pool: OnceLock::new(),
        })
    }

    fn ensure(&self) -> Result<&rayon::ThreadPool, String> {
        if let Some(pool) = self.pool.get() {
            return Ok(pool);
        }
        let built = rayon::ThreadPoolBuilder::new()
            .num_threads(self.n_threads)
            .build()
            .map_err(|err| format!("failed to build E-step rayon pool: {err}"))?;
        #[cfg(test)]
        POOL_BUILDS_THIS_THREAD.with(|builds| builds.set(builds.get().saturating_add(1)));
        let _ = self.pool.set(built);
        self.pool
            .get()
            .ok_or_else(|| "E-step rayon pool missing after build".to_string())
    }
}

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

/// How many chunks have a non-empty person range.
///
/// Empty chunks are a trailing suffix of the `ceil` partition, so this is
/// `O(1)` and does not walk `n_chunks` when that count exceeds `n_persons`.
pub(crate) fn nonempty_person_chunk_count(n_persons: usize, n_chunks: usize) -> usize {
    if n_persons == 0 || n_chunks == 0 {
        return 0;
    }
    let chunk_size = n_persons.div_ceil(n_chunks);
    let last = (n_persons - 1) / chunk_size;
    (last + 1).min(n_chunks)
}

fn map_live_window<T, F>(
    pool: &PersonChunkPool,
    n_persons: usize,
    n_chunks: usize,
    offset: usize,
    end: usize,
    map_chunk: &F,
) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    if end - offset == 1 || pool.n_threads == 1 {
        let mut out = Vec::with_capacity(end - offset);
        for chunk_idx in offset..end {
            let (start, stop) = person_chunk_range(n_persons, n_chunks, chunk_idx);
            out.push(map_chunk(start, stop));
        }
        return Ok(out);
    }
    let thread_pool = pool.ensure()?;
    Ok(thread_pool.install(|| {
        (offset..end)
            .into_par_iter()
            .map(|chunk_idx| {
                let (start, stop) = person_chunk_range(n_persons, n_chunks, chunk_idx);
                map_chunk(start, stop)
            })
            .collect::<Vec<_>>()
    }))
}

/// Map each non-empty person chunk and fold in chunk-index order.
///
/// At most `n_threads` partials exist at once. The fold itself stays a
/// left fold over ascending chunk index, so the association tree does not
/// depend on the window width. `map_chunk` is not called for empty ranges.
pub(crate) fn fold_person_chunks<T, F, A, R>(
    pool: &PersonChunkPool,
    n_persons: usize,
    n_chunks: usize,
    map_chunk: F,
    mut acc: A,
    mut reduce: R,
) -> Result<A, String>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
    R: FnMut(A, T) -> A,
{
    if n_chunks < 1 {
        return Err("e_step_n_chunks must be >= 1".into());
    }
    let n_live = nonempty_person_chunk_count(n_persons, n_chunks);
    if n_live == 0 {
        return Ok(acc);
    }
    let width = pool.n_threads.max(1);
    let mut offset = 0usize;
    while offset < n_live {
        let end = (offset + width).min(n_live);
        let partials = map_live_window(pool, n_persons, n_chunks, offset, end, &map_chunk)?;
        for part in partials {
            acc = reduce(acc, part);
        }
        offset = end;
    }
    Ok(acc)
}

/// Fold non-empty chunks in index order, using the first partial as the
/// accumulator.
///
/// A one-chunk sweep therefore keeps a single counts tensor. `Ok(None)`
/// means there was no non-empty chunk (`n_persons == 0`). Adding a
/// zero-filled tensor into the first partial is not bit-identical work:
/// `0.0 + x == x` for finite `x`, so callers that used to seed with zeros
/// keep the same association tree. Later windows still hold that
/// accumulator plus at most `n_threads` new partials.
pub(crate) fn fold_person_chunks_from_first<T, F, R>(
    pool: &PersonChunkPool,
    n_persons: usize,
    n_chunks: usize,
    map_chunk: F,
    mut reduce: R,
) -> Result<Option<T>, String>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
    R: FnMut(T, T) -> T,
{
    if n_chunks < 1 {
        return Err("e_step_n_chunks must be >= 1".into());
    }
    let n_live = nonempty_person_chunk_count(n_persons, n_chunks);
    if n_live == 0 {
        return Ok(None);
    }
    let width = pool.n_threads.max(1);
    let mut offset = 0usize;
    let end = width.min(n_live);
    let mut partials = map_live_window(pool, n_persons, n_chunks, offset, end, &map_chunk)?;
    let mut acc = partials.remove(0);
    for part in partials {
        acc = reduce(acc, part);
    }
    offset = end;
    while offset < n_live {
        let end = (offset + width).min(n_live);
        let partials = map_live_window(pool, n_persons, n_chunks, offset, end, &map_chunk)?;
        for part in partials {
            acc = reduce(acc, part);
        }
        offset = end;
    }
    Ok(Some(acc))
}

/// Collect non-empty chunk partials in chunk-index order.
///
/// Prefer [`fold_person_chunks_from_first`] in production sweeps so the
/// counts tensors are reduced as each window finishes. This collector
/// exists for tests that compare partials directly.
#[cfg(test)]
pub(crate) fn map_person_chunks<T, F>(
    pool: &PersonChunkPool,
    n_persons: usize,
    n_chunks: usize,
    map_chunk: F,
) -> Result<Vec<T>, String>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    fold_person_chunks(
        pool,
        n_persons,
        n_chunks,
        map_chunk,
        Vec::new(),
        |mut acc, part| {
            acc.push(part);
            acc
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

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
        assert_eq!(nonempty_person_chunk_count(3, 5), 3);
        assert_eq!(nonempty_person_chunk_count(10, 6), 5);
        assert_eq!(nonempty_person_chunk_count(0, 4), 0);
    }

    #[test]
    fn ordered_map_is_bit_identical_across_thread_counts() {
        let n_persons = 17usize;
        let n_chunks = 4usize;
        let map = |start: usize, end: usize| -> f64 {
            (start..end).map(|p| (p as f64 + 1.0).sin()).sum()
        };
        let pool_1 = PersonChunkPool::new(1).unwrap();
        let pool_8 = PersonChunkPool::new(8).unwrap();
        let a = map_person_chunks(&pool_1, n_persons, n_chunks, map).unwrap();
        let b = map_person_chunks(&pool_8, n_persons, n_chunks, map).unwrap();
        assert_eq!(a.len(), nonempty_person_chunk_count(n_persons, n_chunks));
        assert_eq!(b.len(), a.len());
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

    #[test]
    fn empty_chunks_are_not_mapped() {
        let calls = AtomicUsize::new(0);
        let pool = PersonChunkPool::new(2).unwrap();
        let parts = map_person_chunks(&pool, 3, 20, |start, end| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert!(start < end, "empty range must not be mapped");
            end - start
        })
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(parts, vec![1, 1, 1]);
    }

    #[test]
    fn in_flight_partials_stay_within_thread_budget() {
        let in_flight = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        let pool = PersonChunkPool::new(2).unwrap();
        let _ = map_person_chunks(&pool, 4, 12, |_start, _end| {
            let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            thread::sleep(Duration::from_millis(40));
            in_flight.fetch_sub(1, Ordering::SeqCst);
            1u8
        })
        .unwrap();
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak in-flight {} exceeded thread budget 2",
            peak.load(Ordering::SeqCst)
        );
        assert_eq!(in_flight.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn one_pool_serves_repeated_parallel_folds() {
        reset_pool_builds_this_thread();
        let pool = PersonChunkPool::new(3).unwrap();
        let map = |start: usize, end: usize| (start..end).map(|p| p as f64).sum::<f64>();
        let once = fold_person_chunks(&pool, 9, 4, map, 0.0f64, |acc, part| acc + part).unwrap();
        let twice = fold_person_chunks(&pool, 9, 4, map, 0.0f64, |acc, part| acc + part).unwrap();
        assert_eq!(once.to_bits(), twice.to_bits());
        assert_eq!(pool_builds_this_thread(), 1);
    }

    #[test]
    fn fold_from_first_matches_zero_seed_and_skips_empty_input() {
        let pool = PersonChunkPool::new(3).unwrap();
        let map = |start: usize, end: usize| {
            (start..end).map(|p| (p as f64 + 0.5).sin()).sum::<f64>()
        };
        let seeded = fold_person_chunks_from_first(&pool, 9, 4, map, |acc, part| acc + part)
            .unwrap()
            .expect("nine persons produce a partial");
        let zeroed = fold_person_chunks(&pool, 9, 4, map, 0.0f64, |acc, part| acc + part).unwrap();
        assert_eq!(seeded.to_bits(), zeroed.to_bits());
        let serial = PersonChunkPool::new(1).unwrap();
        let one_chunk = fold_person_chunks_from_first(&serial, 9, 1, map, |acc, part| acc + part)
            .unwrap()
            .expect("one chunk");
        let one_zero =
            fold_person_chunks(&serial, 9, 1, map, 0.0f64, |acc, part| acc + part).unwrap();
        assert_eq!(one_chunk.to_bits(), one_zero.to_bits());
        assert!(fold_person_chunks_from_first(&pool, 0, 4, map, |acc, part| acc + part)
            .unwrap()
            .is_none());
    }
}
