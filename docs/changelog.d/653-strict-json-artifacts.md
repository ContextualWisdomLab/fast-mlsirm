# Strict JSON artifact interoperability

## Fixed

- Governed JSON artifact writers now reject `NaN`, positive infinity, and negative infinity instead of emitting Python's non-standard JSON numeric extensions, preserving RFC 8259 interoperability and atomic publication failure.
- Non-finite serialization errors use a bounded package-owned message without reflecting the rejected artifact payload.
