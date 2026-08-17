# Relation-safe structural model comparison contract

## Added

- Added a typed structural measurement-model relation contract that keeps factor
  retention separate from structural model choice and classifies model pairs
  from explicit parameter-space, boundary, constraint, overlap, and formal
  distinguishability facts rather than model names.
- Restricted regular likelihood-ratio procedures to regular nesting, routed
  boundary/unidentified/nonlinear restrictions to conservative bootstrap LR,
  required formal Vuong distinguishability before non-nested selection, and
  returned explicit no-selection or unknown states instead of forcing a winner.
- Added fail-closed contradiction, exact-Boolean, boundary-precedence, and
  procedure-routing tests plus APA 7 doctoring; no comparison statistic or
  estimator is introduced by this contract slice.
