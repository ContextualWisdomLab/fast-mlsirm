# Requirements, decisions, implementation and evidence matrix

Status: **Authoritative traceability baseline**  
Last reviewed: 2026-08-09

This matrix makes the major product requirements discoverable without reconstructing decisions from chat history or PR bodies. It deliberately distinguishes **protected-main implementation**, **open/proposed work**, and **future research**.

| Requirement family | PRD / TRD IDs | ADR | Protected-main implementation/evidence | State |
|---|---|---|---|---|
| Repository ownership | PRD-PRN-007, TRD-BOUND-001/002 | ADR-0001 | `AGENTS.md`, `CLAUDE.md`; package boundary in `python/fast_mlsirm/` | Accepted |
| Rust numerical ownership | PRD-PRN-002, TRD-NUM-001..006 | ADR-0002 | `crates/mlsirm-core/`, `crates/fast-mlsirm-py/`, backend/parity tests | Accepted |
| Assessment/scoring contracts | PRD-FR-001..004, TRD-API/P-ROV/SCR | ADR-0003, ADR-0005 | `python/fast_mlsirm/scoring/contracts.py` and bounded submodules | Accepted |
| Rubric/blueprint/generation | PRD-FR-010..014, TRD-RUB-001..006 | ADR-0003, ADR-0004 | `python/fast_mlsirm/rubric/`: models/compiler/contracts/generation/candidates/audit/pilot modules | Partial / evolving |
| Governed item bank lifecycle | PRD-FR-010, FR-080 | ADR-0004 | Pilot/admission/lifecycle primitives exist; complete approved-bank monitoring/retirement workflow remains evolving | Proposed/partial |
| Automated essay scoring | PRD-FR-020..023, TRD-SCR | ADR-0005 | governed essay score, calibration, validation and HTML report modules/tests from v0.7.0-era work | Accepted baseline / evolving diagnostics |
| Enterprise issue evaluation | PRD-FR-020..023 | ADR-0005 | `fast_mlsirm.scoring.enterprise_issue` adapters, governed observations/calibration/reporting | Accepted reusable adapter |
| Reference-free RAG measurement | PRD-FR-030..033, TRD-RAG | ADR-0005, ADR-0006 | research/design direction exists; canonical full RAG observation pipeline is not yet an accepted protected-main end-to-end feature | Proposed |
| Model relation/comparison | PRD-FR-040..043, TRD-MOD | ADR-0006 | relation-safe comparison primitives and diagnostics where merged; formal family-wide distinguishability remains work in progress | Partial |
| Bifactor scoreability | PRD-FR-044, TRD-BIF | ADR-0006 | Rust/Python feature work exists in repository history/open work; exact released capability must be checked against current public exports before use | Evolving |
| Factor retention | PRD-FR-040/050, TRD-MOD-001 | ADR-0006 | dimensionality diagnostics exist; unified factor-retention evidence API remains a product gap | Partial |
| Adaptive rotation | PRD-FR-051/052, TRD-ROT | ADR-0009 | substantial implementation exists on Draft/open work, not an Accepted protected-main architecture yet | Proposed |
| True-parameter recovery | PRD-PRN-003, TRD-TEST-003..006 | ADR-0008 | simulation/recovery reports, Rust/NumPy parity, scheduled statistical studies/recovery contracts | Accepted |
| Multilevel/multiple-membership/temporal | PRD-FR-060..062, TRD-MLT | ADR-0007 | contextual summaries exist; full reusable contract PR remains open and Rust estimator recovery is future work | Proposed/partial |
| Accessible standalone reports | PRD-FR-070..072, NFR-004 | ADR-0005 | report renderers, exact-value exports, WCAG-focused regression/doctoring | Accepted/evolving |
| LLM credentials/orchestration | TRD-LLM-001..004 | ADR-0010 | repo/org automation contracts; deterministic paths avoid unnecessary model credentials | Accepted governance |
| Release/provenance | PRD-FR-080..082, TRD release section | ADR-0003 | release acceptance, commercial evidence, buyer packet, SBOM/provenance/readiness builders | Accepted baseline |

## Key source locations

### Canonical public contract composition

- `python/fast_mlsirm/scoring/contracts.py`
- `python/fast_mlsirm/rubric/__init__.py`
- `python/fast_mlsirm/__init__.py`

### Numerical source of truth

- `crates/mlsirm-core/src/`
- `crates/fast-mlsirm-py/src/`

### Scientific and product evidence

- `tests/`
- `fuzz/`
- `docs/doctoring/`
- `docs/changelog.d/`
- release/recovery/governance scripts under `scripts/`

## Documentation authority

The documentation authority order is:

1. protected-main source and tests for executable behavior;
2. accepted ADRs for governing architectural/scientific decisions;
3. `docs/PRD.md` and `docs/TRD.md` for product/technical requirements;
4. root `ARCHITECTURE.md` and diagram sources for system views;
5. method-specific/doctoring documentation and primary literature;
6. proposed ADRs/open issues/PRs for future work.

PR bodies, automation handoffs and conversations are evidence/discovery inputs but are not authoritative after their decisions have been captured here.

## Maintenance rule

A PR that materially changes a public contract, bounded-context ownership, numerical owner, model interpretation, lifecycle, trust boundary, or release criterion must update this matrix or explicitly demonstrate that the existing mapping remains correct.
