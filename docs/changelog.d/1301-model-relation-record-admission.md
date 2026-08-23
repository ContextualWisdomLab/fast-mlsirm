# Model-relation record admission

## Fixed

- Reject caller-defined `ModelRelationEvidence` subclasses before structural-relation fields are read, preventing caller callbacks from altering the evidence used to choose LR, bootstrap-LR, or Vuong procedures.
- Replay package-owned relation-evidence validation before classification so post-construction frozen-record mutation cannot inject callback-bearing or contradictory structural facts while preserving valid exact-record behavior.
