# Confirmatory loading-pattern resource envelope

## Fixed

- Bound confirmatory loading-pattern admission and replay to the existing package-owned `MAX_IRT_RESPONSE_CELLS` logical-cell envelope before full finite/binary scans or dense `int64` materialization. Exact NumPy broadcast/strided views and exact list/tuple matrices that exceed the envelope now fail closed while admissible loading structures retain their existing shape, binary, immutability, and confirmatory-model semantics.
- Canonical loading evidence is now materialized as an exact C-contiguous `int64` NumPy view over immutable package-created byte storage. Public callers cannot re-enable `WRITEABLE` and mutate the constructor-created loading pattern in place, and replay requires that concrete immutable backing contract before binary validation. Read-only views over caller-owned mutable arrays continue to fail closed; `OWNDATA=True` is no longer treated as an immutability/provenance guarantee.
