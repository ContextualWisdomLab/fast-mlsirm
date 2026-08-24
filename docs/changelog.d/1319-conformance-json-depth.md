# Bound conformance-manifest JSON nesting before decoding

## Fixed

- Reject cross-engine conformance manifest JSON whose structural nesting exceeds the package replay budget before `json.loads` can enter recursive decoding.
- Preserve the exact existing nesting boundary while ignoring bracket-like characters inside quoted JSON strings.
- Add focused regressions proving the over-budget failure occurs before decoder execution and that the exact nesting budget remains admissible to the raw preflight.
