# Keep judge runtime validation active under Python optimization

## Fixed

- Replace production judge and calibration invariants that relied on removable `assert` statements with explicit package-owned `ValueError` or `RuntimeError` failures, and verify that invalid response-schema admission remains fail-closed under `python -O`.
