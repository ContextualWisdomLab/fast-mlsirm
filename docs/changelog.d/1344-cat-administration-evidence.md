# Seal dichotomous CAT administration evidence

## Fixed

- Reject callback-bearing top-level array providers, ndarray/container subclasses, and non-real storage for partial CAT administered-item and response evidence before NumPy materialization or Rust ability-estimation dispatch.
- Preserve exact NumPy and ordinary built-in list/tuple numeric evidence, including concrete NumPy scalar compatibility, while retaining lossless signed-64 item-index validation, item range/uniqueness rules, and the exact 0/1 response contract.
- Reject over-rank, length-mismatched, and structurally impossible partial administrations from inert container metadata before value-wise scans or dense `int64`/`float64` marshalling; a partial administration cannot exceed the calibrated bank item count because administered identities must be unique.
- Apply the same over-bank administration preflight to `ability_standard_error` on both the module and top-level package surfaces before signed-index normalization, uniqueness work, or compiled-core discovery.
- Keep CAT probability, likelihood, EAP/MLE posterior/scoring, Fisher-information selection, stopping, and uncertainty arithmetic Rust-owned; this change is Python validation, bounded materialization, and marshalling only.
