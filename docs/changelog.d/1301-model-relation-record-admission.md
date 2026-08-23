# Model-relation record admission

## Fixed

- Reject caller-defined `ModelRelationEvidence` subclasses before structural-relation fields are read, preventing caller callbacks from altering the evidence used to choose LR, bootstrap-LR, or Vuong procedures while preserving exact package-owned evidence behavior.
