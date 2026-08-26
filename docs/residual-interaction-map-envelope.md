# Residual Interaction-Map Envelope

`fast_mlsirm.residual_interaction_map_envelope(...)` is the versioned, product-neutral persistence boundary for the Rust-owned residual interaction map. It is intended for downstream products that must store or render the map without recomputing rank, coverage, cell selection, residuals, distances, or geometry.

## Scientific and ownership boundary

The numerical map remains the Gabriel symmetric factorization implemented by `mlsirm-core`. The caller supplies observed values, model expectations, opaque person/item identifiers, and an explicit axis count. Python validates and losslessly marshals these inputs; Rust owns complete-case selection and all persisted numerical results.

The public schema is currently:

```text
fast-mlsirm.residual-interaction-map.v1
```

Unsupported schema versions fail closed. The schema identifier is a compatibility contract, not a claim that future major versions are backward compatible.

## Inputs

- `observed`: a two-dimensional real-numeric matrix. `NaN` is the only missing-value representation; infinity is invalid.
- `expected`: a same-shaped two-dimensional real-numeric matrix. Every value must be finite.
- `person_ids`: an exact built-in `list` or `tuple` of unique exact strings, one per observed row.
- `item_ids`: an exact built-in `list` or `tuple` of unique exact strings, one per observed column.
- `axis_count`: an explicit positive integer request validated by the package resource envelope.
- `schema_version`: the exact supported schema above unless a future version is explicitly implemented.

Opaque identifiers are returned unchanged for retained persons/items and deterministic closest/farthest cells. They are identities only: the numerical core does not parse domain meaning from them.

## Cryptographic input provenance

After caller evidence has passed the public validation and lossless binary64 boundary, Python derives a SHA-256 digest over one unambiguous tagged byte stream. The stream includes the exact schema identity, requested axis count, ordered person identifiers, ordered item identifiers, matrix shape, and the validated observed/expected matrices encoded as C-order little-endian float64 bytes. Every tag and payload is length-prefixed, so concatenation cannot create an ambiguous identity.

This digest is provenance serialization, not psychometric arithmetic. The exact lowercase 64-hex digest is passed through the PyO3 boundary. Rust validates its SHA-256 identity format before identifier or numerical validation, stores it in the versioned envelope, and returns it with the Rust-owned result. The public Python binding then requires the returned digest to equal the digest of the current validated request before any numerical payload marshalling. A stale or foreign native result with otherwise current-looking metadata therefore fails closed on input identity mismatch.

Equivalent admitted NumPy and built-in carriers that normalize to the same ordered identities and float64 evidence have the same digest. Changing a validated matrix value, opaque identifier, schema, shape, or requested axis count changes the digest.

## Returned evidence

`ResidualInteractionMapEnvelope` returns:

- schema, algorithm, implementation, calculation-provenance, and SHA-256 input-digest identifiers;
- the caller-requested axis count and the explicit closest/farthest tie policy;
- finite-value status after Rust validates all persisted numerical values;
- retained person/item opaque identifiers and original indices;
- scored, complete-case map, and incomplete person/item counts;
- effective numerical rank;
- deterministic closest/farthest cells and their opaque identifiers;
- person/item coordinates, singular values, and Gabriel inertia shares;
- retained observed and expected values;
- residual, requested-axis distance, reconstruction, unexplained residual, and cross-share evidence.

The deterministic extrema tie rule is:

```text
lexicographic-first-original-index
```

When several retained cells have the same minimum or maximum requested-axis distance, Rust returns the lexicographically first original `(person_index, item_index)`.

## Complete-case and rank-zero behavior

Observed `NaN` cells participate in the existing complete-case policy; they are never filled with zero. If no complete-case rectangle remains, the envelope reports rank 0, zero map persons/items, empty retained identifiers and map arrays, and no closest/farthest cell.

A non-empty complete-case matrix may also have effective rank 0, for example when every centered residual is zero. In that case coordinates and distances remain finite, inertia shares are zero, and deterministic cell extrema still follow the declared tie rule.

## Downstream rule

A downstream consumer may validate the schema, cryptographic input identity, and provenance, persist the returned values, and render them. It must not recompute selection, effective rank, coverage differences, cell extrema, residuals, requested-axis distances, coordinates, reconstruction, or cross-share and then present those derived values as the `fast-mlsirm` result.

## Research basis

Gabriel, K. R. (1971). The biplot graphic display of matrices with application to principal component analysis. *Biometrika, 58*(3), 453–467. https://doi.org/10.1093/biomet/58.3.453

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5