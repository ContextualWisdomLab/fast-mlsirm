# Buyer-packet repository source identity

`build_buyer_packet` now rejects source-bound procurement packets when any repository-owned product document or product manifest differs from the advertised Git `HEAD`. This closes the gap where a clean `source_commit` value could be paired with modified, staged, untracked, or ignored working-tree evidence and then archived as if those bytes came from that revision.

The focused regression first demonstrated the defect by committing the product evidence, modifying `README.md` without committing it, and expecting packet admission to fail. The causal repair validates the fixed repository evidence paths with Git porcelain status, including ignored/untracked entries, before the canonical collector reads them. This validation is confined to the buyer-evidence boundary; LSIRM/MLSIRM/IRT likelihood, estimation, scoring, recovery, simulation, GPU arithmetic, and public numerical semantics are unchanged.

Existing packet fixtures now commit repository-owned evidence and carry the source identities already required by sales-readiness and optional benchmark/release manifests, so the tests exercise the intended source-bound contract rather than an obsolete unbound fixture.
