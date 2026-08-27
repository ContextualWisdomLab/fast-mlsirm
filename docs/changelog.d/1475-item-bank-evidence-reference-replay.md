# Item-bank evidence reference replay

## Fixed

- Replay exact package-owned item-bank evidence-reference identity before standalone serialization so post-construction mutation cannot publish invalid provenance or dispatch caller string protocols.
- Replay every newly supplied lifecycle-transition evidence reference before normalization/successor construction so mutated kind, identifier, or SHA-256 state cannot acquire authoritative item-bank provenance.
