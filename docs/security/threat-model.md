# fast-mlsirm reusable-component threat model

Status: **Authoritative reusable-core threat model**  
Last reviewed: 2026-08-09

This document models threats owned by the reusable `fast-mlsirm` package and its CI/release boundaries. It deliberately does **not** duplicate hosted-product threats for HTTP endpoints, sessions, consent, tenant/RBAC administration, product databases, billing, customer data-rights workflows or deployment control planes; those belong to Psychometrics Commons or the service that owns them.

## 1. Assets and trust boundaries

### Protected assets

- exact assessment/rubric/scoring contract identity and provenance;
- unmodified item/evidence/calibration/model/recovery artifacts;
- psychometric formula/parameterization/gradient semantics;
- exact numerical output, convergence and uncertainty evidence;
- model relation, scoreability and validity boundaries;
- Python↔PyO3↔Rust memory/shape/type contracts;
- provider/model credentials and reviewer/merge credentials as **separate authorities**;
- CI/release source identity, SBOM/provenance and exact tested artifact;
- benchmark/calibration bank confidentiality where required;
- sensitive source/evidence data only to the extent the reusable computation is explicitly authorized to receive it.

### Principal trust boundaries

```text
caller Python objects
   -> bounded Python validation
   -> PyO3/numpy native boundary
   -> Rust numerical core

rubric/blueprint
   -> generation request
   -> external/untrusted provider output
   -> strict deterministic parser
   -> semantic screening
   -> psychometric pilot/calibration

PR source
   -> CI/security/scientific evidence
   -> independent review/branch protection
   -> package/release artifact
```

## 2. Threat inventory and controls

| Threat | Failure mode | Required control | Evidence / recovery |
|---|---|---|---|
| Untrusted JSON/member ambiguity | duplicate keys, NaN/Infinity, unknown fields, deep/oversized payload create parser disagreement | strict bounded JSON, duplicate-member rejection, finite numbers, closed schemas, depth/count/bytes ceilings | hostile parser tests; reject before candidate construction |
| Provider replay/provenance substitution | output for rubric/blueprint/request A is rebound to B | complete content fingerprints, exact echoed identities, source/cardinality checks, recomputed candidate/execution identities | replay/forgery tests; regenerate as new identity |
| Evidence/source fabrication | provider cites undeclared source/span or a span absent from source | exact source ids/digests, bounded verbatim-span validation, later semantic entailment screening | reject structurally; quarantine semantically invalid candidate |
| Benchmark contamination / double dipping | a candidate response defines the rubric used to score itself | candidate-blind benchmark generation; candidate-aware discovery only with separated cross-fit discovery/scoring evidence | fold-isolation tests and separate bank provenance |
| Package artifact mutation | built wheel/sdist/SBOM/provenance no longer corresponds to tested head | exact-head build identity, immutable artifact digests, SBOM/attestation/release-acceptance verification | abort release; rebuild from protected head |
| Self-modifying CI / source laundering | PR-controlled workflow rewrites source and pushes the implementation it claims to test | CI validates reviewed source only; no branch-local self-removing/encoded-patch writer for scientific implementation; least privilege | source-writer absence/permissions tests; close unsafe workflow path |
| PyO3/native shape/type confusion | malformed dimensions/types reach unsafe/native arithmetic or wrong marshalling | bounded Python validation plus Rust validation at native trust boundary; checked products before allocation; typed results | Python/Rust hostile input tests, Miri/fuzz where useful, fail before computation |
| Numeric overflow/non-finite output | dimension products, exponentials, likelihoods or diagnostics overflow and silently return misleading values | checked integer/byte products, finite input/intermediate/output contracts, stable log-domain/scaled algorithms where method permits | boundary/property tests; fail closed with bounded diagnostic |
| CPU oversubscription/resource exhaustion | nested thread pools, enormous workspaces or unconstrained starts/studies starve host/CI | coarse-grained Rust parallelism, explicit worker/batch/workspace ceilings, no unbounded iterable materialization, separate heavy studies from PR smoke | resource-bound tests/metrics; reject oversized workload before allocation |
| GPU evidence spoofing | CPU/software fallback is reported as GPU success | backend/device identity in evidence; no-skip device test; CPU/GPU result parity with declared tolerance | fail GPU claim when device kernel did not execute |
| Scientific model misuse | flexible model fit is represented as validity/scoreability or correlation as recovery | relation-aware comparison, identification/recovery/coverage, scoreability, DIF/invariance/local-dependence and interpretation gates | return indeterminate/not-scoreable instead of preferred/valid |
| Rater/judge authority confusion | LLM/human score treated as truth or model can approve its own PR | fallible-rater contracts; reviewer/merge credentials separate from model credentials; independent review policy | many-facet/agreement/drift evidence; repository protection remains authoritative |
| Credential cross-contamination | provider subprocess receives repo-write/reviewer/OIDC secrets not needed for generation | explicit secret allowlist, stripped child environments, NVIDIA NIM provider credential separated from reviewer/merge authority | workflow/provider environment tests; fail closed when required model secret absent |
| Privacy overcollection | raw PII/source text replicated into durable artifacts/logs/provider errors | purpose limitation, data minimization, opaque references/digests, provider-exception redaction, downstream data-owner authorization/retention | source-free audit tests; revoke source access without mutating non-content provenance where policy allows |
| Blanket masking destroys scientific design | masking prevents longitudinal/context/rater/participant linkage and silently changes estimand | do not substitute masked pseudo-values; use authorized linkage/identity separation and minimum required protected attributes | fail if required authorized linkage unavailable rather than alter design |
| Supply-chain dependency compromise | dependency/action/tool change executes attacker code or changes scientific build | immutable action/source pins where practical, lockfiles, Security Scan/SAST/OSV/SBOM, package/release acceptance | update/remove dependency; narrow documented false-positive suppression only |
| Documentation/decision drift | code and PRD/TRD/ADR/UML/ERD/threat/release docs contradict each other | canonical authority map, status-bearing ADRs, traceability and documentation-contract CI | block release until corrected/superseded |
| Scientific-integrity recovery failure | test threshold, benchmark or model relation is changed after observing an inconvenient result | fail-first tests, prospective recovery targets, exact-head provenance, explicit superseding ADR/doctoring for method change | preserve failed evidence; fix model/design or justify new method independently |

## 3. Abuse cases

### A. Malicious item generator returns plausible but rebound JSON

An external model returns a syntactically valid item but echoes a different rubric/blueprint id and invents a supporting source span. The parser must reject before construction. A provider verdict that it is “valid” has no authority.

### B. Valid structure, invalid meaning

A candidate references a real source span but the span does not entail the keyed answer, or the item is ambiguous. Structural acceptance is insufficient; semantic/content screening quarantines the item before pilot/operational use.

### C. Oversized numerical request

A caller supplies valid-looking arrays whose derived pairwise/workspace dimensions exceed safe memory. Checked size/byte products must reject before node grids, dense distance matrices, or other dominant allocations are created.

### D. Misleading high model fit

A bifactor/latent-space model improves in-sample fit but specific scores are not reliable/recoverable or the extra structure is unsupported out of sample. The system must not turn fit into a released scoring interpretation.

### E. Review model shares repository write authority

A model subprocess is given the same credential used for independent approval/merge and can write its own acceptance evidence. This violates the authority boundary even if the model is trustworthy; model and reviewer/merge credentials must remain separate.

## 4. Privacy and PII strategy

`fast-mlsirm` preserves legitimate analytical linkage while minimizing raw sensitive-content proliferation:

- exact sensitive values are used only for an explicit approved computation;
- identity resolution, customer/participant lifecycle, residency, encryption keys, retention/deletion and data-subject handling remain owned by the downstream data controller/service;
- durable reusable artifacts prefer opaque ids, digests and bounded metadata;
- protected attributes used for DIF/fairness remain governed data and are not casually copied into reports;
- a digest is not assumed anonymous merely because it is not plaintext; and
- absence of authorized linkage is an error when the scientific design requires linkage—it is not repaired by silently flattening or pseudorandom masking.

## 5. Scientific misuse and human governance

The package provides measurement evidence, not autonomous consequential decisions. A downstream system that uses scores for employment, admission, insurance, credit, diagnosis/treatment, discipline, legal rights or other high-impact decisions must establish the appropriate validation, human/governance, authorization, monitoring and legal basis outside this reusable-core threat model.

Enterprise issue priority additionally requires causal outcome/intervention/cost/utility policy; psychometric discrimination is not business/safety criticality.

## 6. Availability and degraded modes

- Provider rate limits/outages block only the provider-backed action; deterministic validation/numerical work continues where possible.
- GPU unavailability can trigger a documented CPU path where the API permits it, but GPU evidence becomes unavailable rather than successful.
- External reviewer/check latency is not a reason to mutate scientific acceptance criteria; unrelated safe work continues.
- A malformed or disconnected measurement design fails before fitting when detectable; the library does not silently coerce it to a simpler estimand.

## 7. Security maintenance gate

Material PRs must update this threat model when they introduce a new trust boundary, persistence/credential authority, native execution surface, artifact mutation path, provider/evidence flow, or scientific interpretation that changes the abuse cases above. Hosted product threats are linked rather than duplicated.

## 8. Standards and evidence basis

The architecture maps to the repository's current standards/research basis in `docs/traceability/research-basis.md`, including secure software development, AI risk/governance, architecture description, requirements engineering and testing/measurement standards. This document supports SOC 2/CSAP readiness evidence but does not claim certification.
