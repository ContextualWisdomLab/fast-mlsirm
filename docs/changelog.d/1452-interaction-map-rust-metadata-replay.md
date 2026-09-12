# Interaction-map Rust metadata replay

## Fixed

- Fail closed before numerical payload marshalling when the Rust interaction-map envelope returns a foreign or missing schema version, algorithm identity, calculation provenance, requested-axis count, extrema tie policy, finite-value status, or implementation version. The public Python boundary now replays the validated v1 metadata contract instead of coercing and exposing stale/foreign extension metadata.
