# contextual-orchestrator credential ownership

## Changed

- Supersede the legacy direct-provider credential policy for `fast-mlsirm`: model-backed GitHub Actions use the centrally owned contextual-orchestrator `orchestrator/free` gateway and gateway token, leaf workflows do not select provider/model/group/paid fallback, and new direct client/schema adoption requires an immutable released upstream contract rather than mutable source.
