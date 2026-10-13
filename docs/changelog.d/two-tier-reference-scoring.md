# Two-tier reference-metric person scoring

## Added

- An orthogonal-primary two-tier GRM fit can now be scored on the reference
  item metric: each person keeps the fit's primary EAPs and gets the
  expected raw total from the shared two-tier expected-raw scorer, together
  with the mean, second moment and variance of the anchor-only expected total
  under a caller-chosen reference prior on the primaries. No latent rescaling
  is done.
- A JSON-serializable payload entry point accepts the same fit as a field
  mapping and returns the scores and reference moments.
- The reference-moment integral runs in the Rust core on the same
  expected-raw computation as per-person scoring; fits estimated with
  correlated primaries are rejected because they need a different primary
  integral.
