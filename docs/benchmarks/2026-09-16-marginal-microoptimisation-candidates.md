# Marginal-estimator micro-optimisation candidates — 2026-09-16

Status: environment-specific timing evidence for open pull requests against
`python/fast_mlsirm/estimators/marginal.py`; not a universal speed claim and not
a review verdict.

Written because eight open pull requests propose changes to two lines of that
file, and the per-cluster choice between them had been argued from descriptions
rather than from measurement at the shapes the code actually runs. The results
below decided two of those choices. They are recorded here rather than left in
pull-request comments so a later reader can check whether they still hold.

All timings are `timeit` on one machine, single run, with the repository's own
pinned NumPy in an isolated environment resolving the Rust core as the default
backend. Arrays from `numpy.random.default_rng(0)`. Ratios are indicative of
direction and magnitude; they are not precise figures and were not repeated
across machines or runs.

## Euclidean distance in the LSIRM M-step

Candidates replace

```python
dist = np.sqrt(eps_distance + np.sum(diff * diff, axis=1))
```

with `np.einsum("ij,ij->i", diff, diff)`. Six open pull requests propose the
identical replacement: #1822, #1825, #1841, #1878, #1897, #1901.

**The shape matters and the proposals did not state it.** At this call site
`diff` is `(Nx, K)` where `K` is `latent_dim`, typically 2 or 3 — a very narrow
array. A speedup measured on a large one would not transfer.

| shape | `np.sum(d*d, axis=1)` | `np.einsum("ij,ij->i", d, d)` | ratio |
| --- | ---: | ---: | ---: |
| (49, 2) | `0.10824 s` | `0.05829 s` | `1.86x` |
| (49, 3) | `0.14368 s` | `0.07364 s` | `1.95x` |
| (343, 3) | `0.34032 s` | `0.30482 s` | `1.12x` |
| (2401, 2) | `1.11839 s` | `0.47621 s` | `2.35x` |

20,000 calls per cell.

The change is sound at the sizes this code runs. **The absolute saving is about
2.5 microseconds per call** at (49, 2), and the site sits inside a per-item
Newton loop inside the EM iteration, so the total depends entirely on a call
count none of the six proposals supplies. The ratio is the eye-catching number;
the microseconds are the one a reader needs.

## Categorical reduction in the GPCM E-step

Candidates replace

```python
r = np.stack([post[y[:, i] == k].sum(axis=0) for k in range(k_cat)], axis=1)
```

Three open pull requests propose different replacements: #1842, #1866, #1893.

| source | head | expression | best of 3 | agreement |
| --- | --- | --- | ---: | --- |
| current | `ce65339f` | as above | `0.2765 s` | — |
| #1866 | `1e280547` | `np.stack([(y[:, i] == k).astype(post.dtype, copy=False) @ post for k in range(k_cat)], axis=1)` | `0.1907 s` | `3.98e-13` |
| #1893 | `6b29336a` | `((y[:, i, None] == np.arange(k_cat)).astype(post.dtype, copy=False).T @ post).T` | `0.0829 s` | **exact** |
| #1842 | `ad53b67c` | `((y[:, i, None] == k_range).astype(post.dtype, copy=False).T @ post).T` | `0.0813 s` | **exact** |

`(2000 persons, 49 nodes, 5 categories)`, 2,000 calls per cell, three runs.

**Correction (2026-09-16).** An earlier revision of this note reported #1893 at
`0.3404 s` with a `3.69e-13` difference and described its `.T` as a no-op on a
one-dimensional mask. That measured #1893 at commit `98b61cbb`, which used the
#1866 form. The pull request was then updated twice, and its current head
`6b29336a` uses the loop-free form instead. The row above replaces the wrong
one. The `.T` is not a no-op at this head: the mask is two-dimensional.

**Two candidates differ in kind, not degree.** #1866 replaces the inner
reduction but keeps the Python loop over categories — one matrix-vector product
per category. #1893 and #1842 build the indicator matrix once and do a single
matrix-matrix product, removing the loop. That is worth `3.3x` against the
current code where #1866 is worth `1.5x`.

#1893 and #1842 are the same expression. The only difference is that #1842
hoists `k_range = np.arange(k_cat)` out of the loop, worth about `2%` here —
`0.0813 s` against `0.0829 s`, which is within the spread of separate runs and
should not decide between them.

Both loop-free forms are bit-exact against the current code. #1866 differs in
the last digits from accumulating per category; `3.98e-13` is far inside this
repository's `1e-6` parity tolerance and harmless.

## What this does and does not establish

It establishes that both optimisations are real at the shapes in use, that the
loop-free categorical reduction (#1842 and #1893, the same expression) beats the
per-category form on speed and exactness together, and that the distance-line
proposals are interchangeable.

It does not establish an end-to-end fit-time improvement, which would need a
whole-fit benchmark rather than a micro-benchmark, and it does not review any
pull request's other contents. Three of the eight carry unrelated changes
alongside their stated one; that is noted on the pull requests and on the queue
issue, not here.
