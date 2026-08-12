# ATA target-gain Rust ownership

## Changed

- Moved result-affecting capped-shortfall target-information gains for automated
  test assembly from Python/NumPy into a bounded Rust PyO3 kernel.
- Kept Python responsible for validated candidate/content/exposure orchestration
  and deterministic tie breaking while the compiled path owns candidate gain
  arithmetic without candidate-by-point broadcast temporaries.
- Added direct Rust and installed-extension parity/ownership regression evidence.
