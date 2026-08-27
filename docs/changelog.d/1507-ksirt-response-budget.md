# Bounded KSIRT response admission

## Fixed

- Bound KSIRT response evidence to 20,000,000 logical cells before dense `float64` marshalling, using exact NumPy shape metadata or bounded rectangular built-in row metadata without opening caller-owned array/container protocols.
- Bound exact built-in response traversal to 40,000,000 structural nodes so zero-cell empty-row fan-out fails before NumPy materialization or compiled-core discovery while every valid non-empty matrix inside the logical-cell envelope remains admissible.
- Replay the existing minimum `2 persons × 1 item` KSIRT design before dense response conversion once callback-free shape/resource admission has established the dimensions.
- Reject nested built-in response cells and non-scalar NumPy cells during bounded shape preflight so they cannot masquerade as a 2-D persons × items rectangle or reach Rust with mismatched dimensions; logical and structural resource errors retain precedence over deeper shape inspection.
- Preserve finite response-option and bandwidth identity through the Python-to-Rust `f64` boundary: exact built-in/NumPy integer and wider-floating evidence that cannot round-trip through binary64 now fails before compiled-core discovery rather than being silently rounded.
- Preserve Rust ownership of rank-based ordinal ability, Nadaraya–Watson option characteristic curves, kernel evaluation, bandwidth application, and expected-score arithmetic.
