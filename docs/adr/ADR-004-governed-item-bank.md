# ADR-004: Item generation is a governed measurement lifecycle, not a prompt helper

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

The repository historically began after assessment items already existed. Reference-free RAG evaluation, automated scoring, and rubric-centered assessment work require a reproducible upstream path from construct/rubric to candidate item, screening, calibration, operational use, and retirement.

A structurally valid LLM output is not a validated item, and a self-evolving rubric cannot be allowed to change the measurement target while the same candidates are being scored.

## Decision

The canonical lifecycle is:

```text
RubricSpecification
→ BlueprintPlan / ItemBlueprint
→ GenerationContract
→ untrusted generated candidate
→ structural validation
→ semantic/evidence screening
→ artificial-crowd and/or human pilot
→ Rust-backed calibration and diagnostics
→ governed item-bank decision
→ active monitoring
→ suspend / regenerate / retire
```

Benchmark-mode rubric/item generation should be candidate-blind when possible. Candidate-aware criterion discovery must use explicit cross-fitting or a separately governed training/discovery bank.

The logical bank states are:

```text
draft → audited → screened → piloting → calibrated → approved → active
active → suspended → active | retired
any pre-active state → retired when rejected
```

The core owns lifecycle contracts and psychometric evidence, not hosted persistence or workflow UI.

## Consequences

- Provider SDKs remain optional adapters rather than core dependencies.
- Structural parser, semantic/evidence screening, calibration, and operational approval are separate gates.
- Item information, fit, DIF, local dependence, exposure, drift, anchor/linking evidence, and approval history can be retained per version.
- Rubric revisions create new version identities instead of mutating historical measurement definitions.

## Alternatives considered

1. **One-shot LLM `generate_item()`** — rejected because it provides no measurement, calibration, or provenance lifecycle.
2. **Use RAGAS/LLM rubric output directly as score truth** — rejected because the evaluator itself is a fallible measurement instrument.
3. **Self-modifying operational rubric** — rejected for benchmark leakage and loss of scale comparability.

## References

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to evidence-centered design. *ETS Research Report Series, 2003*(1), i–29. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
