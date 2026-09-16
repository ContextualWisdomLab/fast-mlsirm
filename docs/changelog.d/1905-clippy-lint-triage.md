# Clippy lint triage for #1905

## Fixed

- Resolve the 24 deny-level `clippy::erasing_op` errors: every site was
  verified (twice, independently) to be the intentional flat row-major
  index idiom (`0 * stride` keeps rows aligned), with no genuine bug.
  Narrow function-level `#[allow]`s with a one-line reason were added;
  no crate-wide allow. `cargo clippy --workspace --all-targets` exits 0.
- Apply clearly-safe machine lints with no behavior change
  (`map_or` to `is_none_or`/`is_some_and`, `contains`, range-contains,
  `is_multiple_of`, `div_ceil`, needless borrows, `copy_from_slice`,
  NaN-preserving boolean simplification, single-match/collapsible/
  for-values/obfuscated-if rewrites, unnecessary cast, doc-list
  indentation). Lib warnings 380 -> 279.
- Truncate non-representable float digits (`excessive_precision`) in
  quadrature tables and numeric constants. All 116 changed literals
  parse to bit-identical `f64` values (verified programmatically);
  the quadrature tables' shortest-roundtrip claim now holds.
  Lib warnings 279 -> 163.

## Changed

- Documented as intentionally left: `needless_range_loop` (114, all
  `HasPlaceholders`, each needs per-site judgment in numeric kernels),
  `too_many_arguments` (28, public API shape; repo convention is
  per-fn allow), `neg_cmp_op_on_partial_ord` (14, load-bearing NaN
  guards where the lint suggestion would change validation behavior),
  `type_complexity` (5, public signatures), `if_same_then_else`
  (2, intentional degenerate arms with distinct documented reasons).
  Test-target warnings are triaged as follow-up in the issue.
