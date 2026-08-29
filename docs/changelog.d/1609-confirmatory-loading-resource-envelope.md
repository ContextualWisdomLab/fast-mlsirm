### Fixed

- Bound confirmatory loading-pattern admission and replay to the existing package-owned `MAX_IRT_RESPONSE_CELLS` logical-cell envelope before full finite/binary scans or dense `int64` materialization. Exact NumPy broadcast/strided views and exact list/tuple matrices that exceed the envelope now fail closed while admissible loading structures retain their existing shape, binary, immutability, and confirmatory-model semantics.
