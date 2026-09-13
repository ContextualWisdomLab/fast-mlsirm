# Cross-engine conformance run provenance

The inventory contract records capability coverage. This stacked slice adds
optional run-level provenance so a later isolated harness can reproduce and
audit one comparison without placing an external R, Stan, commercial, or
other engine in the package or Rust workspace.

The record binds the harness commit, environment fingerprint, RNG algorithm
and seeds, mapping schema/version/hash, tolerance hash and rationale, raw and
normalized output hashes, and license classification. It stores no raw
responses, participant identifiers, restricted test content, proprietary
binaries, or provider output. A missing optional output artifact remains
explicit as `null` and is not interpreted as a passing comparison.

The record is metadata only. It does not execute a comparison, calculate
RMSE/MAE/discrepancy, perform alignment, or establish construct validity,
fairness, transportability, or high-stakes approval. Those claims require a
separate fixed-parameter or fitted-result harness with independent evidence.

## Research basis (APA 7)

Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies
to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102.
https://doi.org/10.1002/sim.8086

Chalmers, R. P. (2012). mirt: A multidimensional item response theory package
for the R environment. *Journal of Statistical Software, 48*(6), 1–29.
https://doi.org/10.18637/jss.v048.i06
