# Governed item-bank screening evidence

## Added

- Added immutable candidate-screening contracts that bind exact item, rubric,
  blueprint, generation-contract, and screening-policy fingerprints to ten
  required structural/semantic screening dimensions.
- Kept `pass`, `accepted_with_limitation`, and `fail` distinct, with failed
  dimensions blocking piloting and accepted limitations requiring explicit
  provenance instead of being silently converted to success.
- Reused the package's bounded metadata and raw-content rejection boundary; no
  estimator, provider SDK, hosted persistence, or new numerical arithmetic is
  introduced by this slice.
