# DETECT evidence admission hardening

## Fixed

- Reject complex or non-real-numeric DETECT response storage before real-valued marshalling so observed binary evidence cannot be silently projected onto different data.
- Reject complex or non-real-numeric DETECT cluster storage before partition normalization so item-to-dimension labels cannot be silently projected onto a different real partition.
- Reject arbitrary response/cluster array-protocol providers before NumPy materialization, while preserving exact NumPy arrays and plain built-in sequences of trusted real scalar values.
- Reject a self-referential or otherwise cyclic list/tuple response or cluster (for example `a = []; a.append(a)`) before flattening instead of looping until the process is killed; cycle detection tracks only the active ancestor path, so legitimate repeated/shared acyclic rows remain accepted.
- Preserve Rust ownership of conditional-covariance and DETECT index arithmetic; the Python change is limited to validation and marshalling.
