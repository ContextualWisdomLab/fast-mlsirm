# Item-bank DIF applicability evidence

## Fixed

- Calibration transitions accept either DIF evidence or explicit
  `dif_not_applicable` evidence, forbid both at once, and keep other lifecycle
  gates unchanged.
