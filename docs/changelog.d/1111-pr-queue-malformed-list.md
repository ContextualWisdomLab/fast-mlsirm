# Fail closed on malformed PR queue evidence

## Fixed

PR queue governance scripts now require the repository-owned bounded JSON helper instead of falling back to a weaker inline decoder, and snapshot capture records malformed-payload errors when otherwise successful open-PR identity or history arrays contain non-object entries while preserving valid records and the raw identity count.
