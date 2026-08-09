# ADR-003: Governed contracts are immutable and content-addressed

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

Assessment specifications, rubrics, task revisions, scoring requests, model/engine identities, generated-item contracts, and observations must be reproducible across local and MSA use. Logical display identifiers alone cannot prove which exact content produced a result, and raw content should not be propagated into every measurement artifact.

## Decision

Governed contracts use deterministic canonical serialization and preserve distinct identity layers:

- logical descriptive identifier;
- schema version;
- semantic/governance version where relevant;
- complete SHA-256 content fingerprint;
- bounded public handle where a compact external reference is useful.

Aggregate factories replay and validate exact package-owned child values before accepting them as authoritative. Post-construction mutation must not silently create a new trusted aggregate identity.

Raw response, source, prompt, provider-output, or PII content is excluded from governed artifacts unless the specific contract requires it. Evidence references store identities/spans/fingerprints rather than implicitly copying source text.

## Consequences

- Version changes invalidate downstream fingerprints deterministically.
- Replays and audits can bind results to exact input revisions.
- Fingerprints are provenance identities only; they are not signatures, encryption, authorization, or anonymity.
- Owning applications may store raw operational content separately under purpose-bound authorization and retention policy.

## Alternatives considered

1. **Logical IDs only** — rejected because content can change beneath the same label.
2. **Store raw content everywhere** — rejected because it increases privacy/security blast radius and couples measurement artifacts to operational storage.
3. **Short hashes as authoritative identity** — rejected; convenience IDs cannot replace full collision-resistant fingerprints for durable provenance.

## References

IETF. (2017). *RFC 8259: The JavaScript Object Notation (JSON) Data Interchange Format*.
