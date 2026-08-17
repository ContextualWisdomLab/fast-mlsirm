# ATA target-gain Rust ownership

## Changed

- Moved result-affecting capped-shortfall target-information gains for automated
  test assembly from Python/NumPy into a bounded Rust PyO3 kernel.
- Kept Python responsible for validated candidate/content/exposure orchestration
  and deterministic tie breaking while the compiled path owns candidate gain
  arithmetic without candidate-by-point broadcast temporaries.
- Made the public PyO3 boundary reject wrong-dtype, non-array, non-contiguous,
  empty-matrix and overlong candidate-set inputs with stable package-owned
  `ValueError` messages before candidate/output allocation.
- Bounded candidate inputs to the item count represented by the information
  matrix and converted both candidate and result vectors with fallible reserve.
- Added direct Rust and installed-extension parity/ownership regression evidence.
