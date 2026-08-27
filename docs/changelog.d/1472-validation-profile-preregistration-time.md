# Validation-profile preregistration chronology

## Fixed

- `ValidationProfile` now records a required, timezone-aware
  `protocol_registered_at` timestamp, normalizes it to UTC, requires it not to
  exceed `analysis_cutoff`, and includes it in canonical serialization and the
  deterministic SHA-256 profile fingerprint. Public identity replay revalidates
  the same chronology after post-construction mutation, while evidence remains
  constrained only by `available_time <= analysis_cutoff` so evidence may
  legitimately predate protocol registration.
