# Confirmatory loading-pattern resource envelope

## Fixed

- Bound confirmatory loading-pattern admission and replay to the existing package-owned `MAX_IRT_RESPONSE_CELLS` logical-cell envelope before full finite/binary scans or dense `int64` materialization. Exact NumPy broadcast/strided views and exact list/tuple matrices that exceed the envelope now fail closed while admissible loading structures retain their existing shape, binary, immutability, and confirmatory-model semantics.
- Replay now also requires the canonical loading array to own its backing storage, so a read-only C-contiguous view over caller-owned mutable memory cannot impersonate package-owned confirmatory evidence after construction.
