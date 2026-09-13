# Lineage channel weight evidence

## Added

- Add a Rust fail-closed evidence contract for continuous lineage channel scores
  and an independently accepted criterion anchor. Weight estimation remains
  unavailable until pair-level independent criterion observations are supplied.

## Changed

- Restore the `mlsirm-core` boundary to a domain-neutral criterion-anchor
  contract. The canonical field is `criterion_anchor`; the historical serialized
  `tepp_anchor` field remains readable only as a compatibility alias. Producer-
  specific schema validation is no longer compiled into the numerical core.
