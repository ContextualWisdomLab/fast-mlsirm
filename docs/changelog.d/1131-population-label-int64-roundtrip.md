# Population-label narrowing safety

## Fixed

- Reject multigroup and multilevel population labels that cannot round-trip through signed 64-bit integer representation before compaction, preventing narrowing overflow from silently reordering the identified reference population while preserving valid sparse labels and the signed `int64` boundary.
