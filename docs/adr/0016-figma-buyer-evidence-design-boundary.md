# ADR-0016: Figma buyer-evidence design boundary

Status: Accepted  
Date: 2026-08-21  
Decision owners: fast-mlsirm maintainers

## Context

The reusable measurement core needs a stable visual contract for buyer-facing
evidence without importing a hosted design system, a Figma runtime, or product
UI code. A design link alone is not sufficient for reproducible review: the
file identity, required review frames, evidence vocabulary, and handoff
boundary must be discoverable from the repository's canonical decision log.

## Decision

The buyer-review design artifact is the following Figma file:

- **Figma File ID:** `qD34PfMH8Kr41tFdqLCkem`
- **Figma URL:** <https://www.figma.com/design/qD34PfMH8Kr41tFdqLCkem>

`fast-mlsirm` owns the repository-local design evidence packet and its
deterministic validation. `scripts/build_figma_evidence_sync.py` validates the
packet's required frame identifiers, buyer-evidence tokens, design URL, and
optional exported metadata snapshot. The packet remains an evidence artifact;
it is not a hosted UI, participant store, or product authorization boundary.

Figma Code Connect remains disabled for this reusable core. Downstream hosted
products may map their own components to the artifact, but that mapping must
not become a runtime or build dependency of `fast-mlsirm`.

## Invariants and acceptance evidence

- `examples/enterprise_demo/figma_design_packet.json` records the artifact URL
  and keeps `code_connect` explicitly false.
- The sync command emits both a machine-readable manifest and an accessible
  HTML report with a restrictive content-security policy.
- Required frame and token omissions fail the sync gate; optional live metadata
  is accepted only when its exported snapshot satisfies the same vocabulary.
- The design boundary is independent of Rust numerical kernels, Python model
  contracts, database ownership, and downstream HTTP/UI code.

## Consequences

Buyers and implementers can locate the exact design source and the next
verification command from one canonical ADR. The repository avoids Figma API
credentials and runtime coupling, at the cost of requiring an exported
metadata snapshot when live-file claims need verification.

## Alternatives considered

- **Undocumented design URL:** rejected because reviewers cannot reliably bind
  a packet to the intended file.
- **Figma Code Connect in the core package:** rejected because it would couple
  a reusable numerical component to a product UI and hosted design tooling.
- **Recreate the hosted product UI here:** rejected by ADR-0001's bounded
  context; product persistence, authorization, and deployment remain
  downstream responsibilities.

## Failure, recovery, and supersession

If the file is renamed, moved, or replaced, update this ADR and the packet in
one reviewed change, then rerun the sync and release-evidence gates. A new
visual source, runtime design integration, or change in repository ownership
requires a superseding ADR rather than silently changing this file.

## References

Figma. (n.d.). *Figma design platform*. Retrieved August 21, 2026, from
<https://help.figma.com/hc/en-us>

