# Cross-engine conformance provenance

## Added

- Add optional run-level conformance provenance for the isolated harness
  commit, environment, RNG seeds, parameter-mapping schema, tolerance
  rationale, output fingerprints, and license classification without storing
  raw responses or adding an external-engine dependency.
- Revalidate exact run-provenance state before direct manifest replay so
  post-construction container rebinding fails closed before caller callbacks.
