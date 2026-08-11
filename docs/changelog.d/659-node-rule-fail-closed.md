# Node-rule fail-closed validation

## Fixed

- Public polytomous and 2PL fitters reject non-string integration-rule controls
  before importing the Rust core, without invoking caller ``__str__``/``__repr__``.
