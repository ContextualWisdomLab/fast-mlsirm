# Oakes factor-id signed-64 admission

## Fixed

- Reject Oakes `factor_id` values that cannot round-trip through signed 64-bit integer marshalling before dimension derivation or Rust uncertainty arithmetic, preventing unsigned overflow from silently changing item-to-dimension assignments.