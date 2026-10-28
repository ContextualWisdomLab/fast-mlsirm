# Stable codebook fingerprints and complete artifact acceptance

## Fixed

- Bind the existing `fast_mlsirm.scale_codebook/1` canonical serialization to
  independent ASCII and Unicode-escape SHA-256 fixtures. Changing serialization
  requires an explicit schema-version and golden-fixture update, not silent
  reuse of the version-1 identity.
- Copy the complete admitted `test_scale_codebook*.py` family into the installed
  wheel acceptance directory and raise the minimum executed test count to 55.
  A repository-level regression prevents the glob and copy command from drifting.
  No statistical formula, study data, or fitted score is changed or included.
