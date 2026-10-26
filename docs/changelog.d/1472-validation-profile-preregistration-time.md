# Validation-profile preregistration chronology

## Fixed

- `ValidationProfile` now records a required, timezone-aware
  `protocol_registered_at` timestamp, normalizes it to UTC, requires it not to
  exceed `analysis_cutoff`, and includes it in canonical serialization and the
  deterministic SHA-256 profile fingerprint. Public identity replay revalidates
  the same chronology after post-construction mutation, while evidence remains
  constrained only by `available_time <= analysis_cutoff` so evidence may
  legitimately predate protocol registration.
- Profile construction now replays each nested evidence `available_time` through
  exact callback-free UTC admission after nested evidence validation and before
  comparison with `analysis_cutoff`. A post-replay mutation therefore fails
  closed instead of executing caller-controlled chronology comparison protocols
  or splicing mixed-time evidence into one preregistered profile.
