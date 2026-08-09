# ADR-007: Humans and LLMs are fallible raters; model execution remains optional

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

Reference-free RAG evaluation, automated essay scoring, and other LLM-as-a-Judge workflows can obtain useful ratings without a single gold answer, but judge outputs contain severity, discrimination, range-use, position/prompt, family, and temporal effects. Human ratings also contain rater error. Treating either source as unqualified ground truth defeats the measurement purpose of this library.

## Decision

Human and automated raters use a shared scoring observation surface while preserving rater/engine provenance. Calibration/validation may estimate or diagnose severity, criterion-specific bias, discrimination, range restriction, agreement, drift, and subgroup effects.

The core package remains provider-neutral and does not hold model credentials. LLM/provider calls belong to optional adapters or an owning orchestration service. Live model-backed tests use `NVIDIA_NIM_API_KEY` from GitHub Secrets; autonomous development must not use `COPILOT_GITHUB_TOKEN`.

When deeper orchestration is scientifically/product-relevant, computation is deliberately allocated across routing, workflow stages, decomposition/recursion, role-specific reasoning effort, and access lists. Single-model and multi-agent designs should be compared under defensible budgets; correctness, evidence quality, and controllability take priority over latency.

## Consequences

- RAGAS/LLM judge scores are observations, not truth labels.
- Raw inter-rater correlation/agreement is descriptive evidence, not sufficient validity evidence.
- `scored`, `abstained`, `failed`, and `excluded` remain distinct.
- Human adjudication remains available for high-uncertainty/high-impact cases.
- Model/provider replacement does not require changing the psychometric core contract.

## Alternatives considered

1. **Single preferred LLM as oracle** — rejected because evaluator bias/drift cannot be distinguished from target quality.
2. **Majority vote as ground truth** — rejected because correlated judge families can share systematic error.
3. **Provider SDK in core** — rejected because it couples scientific release cadence to hosted-model APIs and credentials.

## References

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496.
