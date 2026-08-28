# Count-only compute usage export

## Added

- Add `CanonicalComputeUsageSink`, a provider-neutral simulation/fit metering adapter that exports only bounded compute counts and provenance references through an injected usage-event producer and canonical `usage-event/v1` validator; billing schema, pricing, rating, tax, credit, invoice, persistence, and hosted-product concerns remain downstream-owned.

## Fixed

- Seal metering input, producer-result, validator-result, and validator-to-enqueue trust boundaries with exact built-in carrier/scalar admission. Independent bounded exact-JSON snapshots now prevent successful validator callbacks from rewriting top-level or nested event evidence before durable enqueue; snapshot traversal is limited to 4,096 nodes and depth 16, and callback-bearing nested carriers fail closed before validator or enqueue observation.
