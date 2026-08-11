# ATA semantic-range and exclusion preflight

## Fixed

- Reject negative or contradictory ATA content constraints, invalid exposure-map ranges, negative seeds, and non-integral or out-of-bank exclusions before item-information evaluation while preserving accepted Python/NumPy integer controls and assembly semantics.

## Security

- Keep ATA exclusion identities on an exact package-owned type/range boundary so Boolean, fractional, hostile integer-like, and out-of-bank values cannot be silently coerced or ignored before psychometric work.
