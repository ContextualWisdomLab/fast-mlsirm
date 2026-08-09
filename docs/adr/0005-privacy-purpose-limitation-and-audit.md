# ADR-0005: Purpose-Limited Sensitive Data, Auditability, and Compliance Evidence

- **Status:** accepted
- **Date:** 2026-08-09

## Context

Measurement and automated-scoring systems can require identity-linked or sensitive context to perform legitimate work. Blanket masking can destroy respondent linkage, rater assignment, longitudinal continuity, subgroup/DIF analysis, incident reconstruction, and business workflows. Conversely, copying raw PII into reusable library artifacts, logs, generated-item provenance, or model-provider traces increases breach and compliance risk.

The project is also expected to design toward enterprise assurance programs such as SOC 2 and Korean CSAP without falsely claiming certification.

## Decision

Use **purpose limitation, data minimization, identity separation, scoped authorization, provenance, and auditable access** instead of blanket PII masking as the primary architecture.

Within `fast-mlsirm`:

- prefer opaque identifiers, content digests, source/span references, and source-free derived evidence when raw content is not scientifically required;
- do not retain provider prompts/responses or raw source text in audit artifacts merely for convenience;
- reject sensitive metadata in reusable contracts when the field is not required by the measurement contract;
- keep scientific grouping variables needed for DIF/invariance/multilevel analysis explicit, but do not turn them into identity credentials;
- make terminal/abstained/failed/excluded states explicit rather than encoding absence as a low score;
- keep deterministic provenance sufficient to replay or reject artifact substitution where the contract promises it.

Within the hosted owning product/service:

- identity linkage, SSO/SCIM/RBAC, tenant isolation, encryption keys, data residency, retention/deletion/export, consent, legal basis, privileged access, and break-glass workflows are product responsibilities;
- access to sensitive raw data must be purpose-bound and auditable;
- downstream products decide which sensitive attributes may be passed to the library for an authorized measurement task.

## Compliance posture

The repository may provide control evidence useful for SOC 2/CSAP readiness, but documentation and reports must say **designs toward** or **supports evidence for** those controls unless actual certification/assessment establishes otherwise.

Security gates such as dependency review, SAST, fuzzing, provenance checks, least-privilege workflow permissions, and bounded untrusted-input handling are engineering controls, not certification by themselves.

## LLM/provider boundary

- Model-backed development/evaluation uses the approved secret path (`NVIDIA_NIM_API_KEY` where required) and preferably the owning orchestration service.
- `COPILOT_GITHUB_TOKEN` is not an autonomous development credential.
- Provider exceptions and audit records must avoid echoing raw uncontrolled payloads or credentials.
- Candidate generation/scoring inputs sent externally are governed by the owning product's egress/data policy; the library does not assume external transmission is permitted.

## Alternatives rejected

1. **Mask every PII-like field before measurement.** Rejected because it can destroy legitimate linkage, longitudinal, rater, subgroup, and operational semantics.
2. **Store raw source/provider payloads in every audit artifact.** Rejected because reproducibility can often be achieved with immutable IDs/digests/references while reducing exposure.
3. **Claim certification because CI security checks pass.** Rejected as materially misleading.

## Consequences

- More explicit separation between scientific identifiers and identity credentials.
- Downstream products must implement strong authorization/governance rather than relying on this library as a privacy gateway.
- Some investigations require privileged retrieval of source data from its owning system rather than reading it from the measurement artifact.
- DIF/fairness analysis remains possible because authorized grouping attributes can be supplied deliberately instead of being indiscriminately destroyed.
