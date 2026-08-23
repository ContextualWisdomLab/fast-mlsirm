# Fail closed on malformed PR queue lists

## Fixed

PR queue snapshot capture now records a malformed-payload error when otherwise successful open-PR identity or history arrays contain non-object entries, while preserving useful valid records and the raw identity count for governance evidence.
