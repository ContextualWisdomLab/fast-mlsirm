# Bound PR queue Git metadata lookup

## Fixed

- PR queue governance now bounds the `git rev-parse HEAD` subprocess and fails closed with a stable timeout error instead of allowing a hung local Git child to stall the evidence pipeline.
