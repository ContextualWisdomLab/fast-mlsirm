# Rust authority in release acceptance

## Fixed

- Release acceptance now rejects an automatic fit unless both the persisted fit
  summary and CLI result report the Rust backend. The production acceptance path
  can no longer certify an explicit/reference NumPy numerical owner as the
  automatic backend.
