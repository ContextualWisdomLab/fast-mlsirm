# Fail closed on missing release source identity

## Fixed

- The buyer-facing release evidence index now rejects timed-out, failed, unavailable, empty, malformed, or non-canonical Git `HEAD` identity instead of allowing an otherwise complete packet to report `status: "ok"` with unreconstructable source provenance.
- Valid repositories continue to record the exact full lowercase hexadecimal source commit without changing psychometric/statistical numerical ownership.
