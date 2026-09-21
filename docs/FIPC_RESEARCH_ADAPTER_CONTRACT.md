# FIPC Research Adapter Contract

The research consumer owns row identity and expected-raw evidence. The
`fast-mlsirm` library owns fitting, item parameters, posterior rows, and GPU
receipts. The adapter must not infer either missing field from a permutation
hash or from a fixture target.

## Required Inputs

1. An immutable FIPC artifact produced by the library, including
   `input_focal_sha256`, output field hashes, convergence state, and GPU
   receipt.
2. A `fipc-row-identity/v1` sidecar containing stable `row_ids`, a complete
   zero-based `permutation`, its little-endian int64 SHA-256, and
   `source_input_sha256` equal to the artifact's focal input hash.
3. A `fipc-expected-raw/v1` sidecar containing finite `values`, their
   little-endian float64 SHA-256, `source_input_sha256`, parameter hashes for
   `a_primary`, `a_specific`, `threshold`, and `theta_p_eap`, and a non-empty
   `formula_id` owned by the research consumer.

## Materialization

Run without fitting:

```text
python scripts/fipc_research_adapter_materialize.py \
  artifact.json row_identity.json expected_raw.json derived.json
python scripts/fipc_research_adapter_preflight.py derived.json \
  --expected-source SOURCE_SHA \
  --expected-core CORE_SHA \
  --output preflight.json
```

The materializer fails closed on any mismatch and writes a new derived JSON;
it never edits the input artifact. The preflight must report
`research_consumption_ready=true` before any research result can be consumed.

## Late-Life Owner Commands

The late-life consumer owner must supply the real row IDs and expected-raw
values from the existing research input/checkpoint lineage, compute the hashes
with the specified encodings, run materialization, and attach `preflight.json`
to the release evidence. Do not copy or invent source data on MacBookAir and
do not rerun the fit merely to manufacture these sidecars.
