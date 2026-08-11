# Top-1 CSR input bounds

## Fixed

- Bound top-1 loser streams to at most `n - 1` items, enforce the shared `MAX_RANKING_CSR_BYTES` ceiling on winner/loser/start `uint64` payloads, and normalize ordinary outer/inner iteration failures to stable non-reflective package errors while propagating process-control signals.
