# LSR ranking input bounds

## Standards

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Maydeu-Olivares, A., & Böckenholt, U. (2005). Structural equation modeling of paired-comparison and ranking data. *Psychological Methods, 10*(3), 285–304. https://doi.org/10.1037/1082-989X.10.3.285

## Rationale

Caller-controlled ranking iterables can be infinite or oversized. Before the Rust LSR kernels run, Python materializes CSR arrays of fixed-width `uint64` indices. That handoff must:

1. consume at most `n + 1` entries per ranking (prove overlength without unbounded `list()`);
2. refuse streams that would exceed `MAX_RANKING_CSR_BYTES` of live flat/start payload;
3. redact ordinary iteration failures so hostile payloads never appear in public errors;
4. preserve process-control exceptions.

Numerical Plackett–Luce / LSR arithmetic remains Rust-owned. Bradley–Terry MM
and the additive-ties BRATT variant are separate ranking estimators; see
[`../bradley_terry_mm.md`](../bradley_terry_mm.md) and
[`../adr/0017-bradley-terry-mm.md`](../adr/0017-bradley-terry-mm.md). This
page does not make a Bradley–Terry product claim.

## Implementation

`python/fast_mlsirm/scaling.py` — `MAX_RANKING_CSR_BYTES`, `_rankings_to_csr`.
