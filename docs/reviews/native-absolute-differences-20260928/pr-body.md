## Proposal
Compute per-entry absolute differences and maximum in native core. Require same-length nonempty finite real vectors, reject overflow, no broadcasting. Caller owns identifier alignment and scientific acceptance.

## Source and validation
Docstrings cite actual NumPy v2.5 subtract/absolute/max manual Returns definitions. Native29 tests passed, exit0, with independent NumPy parity and invalid/overflow admission. Study consumer3 tests passed including actual native dispatch of keyed group arrays. JUnit/build/core hashes committed here.

## Remaining acceptance
Current-head required CI, nonauthor review, normal integration and immutable release/install validation. Target base feat/native-nested-subset-20260928. PR creation pending GitHub API quota restoration; do not treat pushed branch as reviewed or released.
