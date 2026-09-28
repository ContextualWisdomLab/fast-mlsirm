# Focal-prior person scoring after 1-D GRM FIPC

## Added

- `score_poly_fipc_group_persons` calibrates one focal group with 1-D GRM
  fixed-item parameter calibration, then returns each person's EAP, posterior
  SD, and plug-in expected raw score on the anchor-fixed reference metric. An
  unconverged calibration raises before any score is produced.

## Changed

- `score_polytomous` accepts an optional Gaussian prior mean and SD. FIPC group
  scoring passes its fitted focal prior to the existing Rust EAP kernel.
