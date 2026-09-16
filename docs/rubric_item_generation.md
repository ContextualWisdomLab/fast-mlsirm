# Rubric-Centered Item Generation

`fast_mlsirm.rubric` turns an approved scoring rubric into reproducible item-authoring contracts before model generation and psychometric calibration. It supplies the construct, evidence, task-model, version, and provenance boundaries that ad hoc prompts usually omit.

## Why this exists

A rubric alone does not define an assessment item. A defensible item-authoring process also needs:

- the construct to be elicited;
- observable evidence for every score level;
- task families capable of eliciting that evidence;
- intended difficulty and evidence conditions;
- constraints that the authoring model must not violate;
- a governed rubric revision independent of the wire-schema version; and
- immutable provenance linking each candidate to the exact rubric and blueprint.

The compiler makes those requirements explicit and content-addressed:

```text
RubricSpecification
        |
        v
BlueprintPlan
        |
        v
ItemBlueprint[]
        |
        v
GenerationContract / canonical JSON / provider-neutral prompt
```

The package does **not** claim that an LLM-generated candidate is valid merely because it conforms structurally. A complete, pilot-eligible `CandidateScreeningResult` is required before `build_pilot_candidate_record` can admit the candidate. Content screening, artificial-crowd trials, human review, and Rust-backed psychometric calibration remain separate quality gates.

## Complete example

```python
from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    build_generation_contract,
    canonical_generation_contract,
    compile_item_blueprints,
    render_generation_prompt,
)

rubric = RubricSpecification(
    rubric_id="faithfulness_rubric",
    construct_id="evidence_grounding",
    construct_definition=(
        "Degree to which every substantive response claim is supported by "
        "the supplied evidence."
    ),
    response_format=ResponseFormat.ORDINAL_RATING,
    levels=(
        RubricLevel(
            score=0,
            label="unsupported",
            descriptor="Substantive claims are not supported.",
            observable_indicators=("unsupported substantive claim",),
        ),
        RubricLevel(
            score=1,
            label="partial_support",
            descriptor="Some but not all substantive claims are supported.",
            observable_indicators=("mixed supported and unsupported claims",),
        ),
        RubricLevel(
            score=2,
            label="full_support",
            descriptor="Every substantive claim is supported.",
            observable_indicators=("complete source support",),
        ),
    ),
    task_families=(
        "claim_verification",
        "citation_attribution",
    ),
    evidence_requirements=(
        "Quote the source span that supports the expected judgment.",
    ),
    prohibited_patterns=(
        "Do not reward fluent wording when source support is absent.",
    ),
    locale="en-US",
    rubric_version="1.0.0",
)

plan = BlueprintPlan(
    difficulty_bands=(DifficultyBand.EASY, DifficultyBand.HARD),
    evidence_modes=(
        EvidenceMode.SINGLE_SOURCE,
        EvidenceMode.MULTI_SOURCE,
    ),
    items_per_cell=2,
    seed=20260803,
)

blueprints = compile_item_blueprints(rubric, plan)
first = blueprints[0]
contract = build_generation_contract(rubric, first)

print(rubric.rubric_version)
print(rubric.fingerprint)
print(first.blueprint_id)
print(contract["blueprint"]["blueprint_handle"])
print(first.blueprint_fingerprint)
print(contract["contract_id"])
print(contract["contract_handle"])
print(contract["contract_fingerprint"])
print(first.generation_seed)
print(canonical_generation_contract(rubric, first))
print(render_generation_prompt(rubric, first))
```

The example combines two task families, two difficulty bands, two evidence modes, and two replicates per design cell. The resulting matrix therefore contains 16 blueprints in a stable declared order.

## Schema and governance versions

`schema_version` and `rubric_version` represent different contracts.

- `schema_version` identifies the serialization contract implemented by the package. This slice accepts `1.0`.
- `rubric_version` identifies the human-governed measurement specification and uses canonical numeric semantic versioning such as `1.2.3`.

A rubric revision changes the rubric fingerprint and therefore changes every downstream blueprint and generation-contract fingerprint. A blueprint compiled under one revision cannot be replayed under another revision, even when its display name appears unchanged.

## Rubric levels

Scores must be contiguous integers beginning at zero. A level includes:

- a concise label;
- a behavioral descriptor; and
- one or more observable indicators.

This prevents a model from receiving only vague adjectives such as “poor,” “average,” and “good” without evidence that can distinguish them.

## Identifiers

All public identifiers and task-family identifiers use two-or-more-token lower `snake_case`:

```text
faithfulness_rubric        valid
claim_verification         valid
rubric                     invalid
Faithfulness_Rubric         invalid
faithfulness-rubric         invalid
```

Three provenance identity layers are deliberately distinct:

- `blueprint_id` and `contract_id` are 16-hex, 64-bit convenience display identifiers retained for compatibility and human-readable logs;
- `blueprint_handle` and `contract_handle` are authoritative 32-hex, 128-bit public handles used across service and audit boundaries; and
- `blueprint_fingerprint` and `contract_fingerprint` are the complete 64-character SHA-256 digests and remain the ultimate content identities stored in durable provenance records.

```text
item_blueprint_6b95a4d2d45a0b42                  convenience id
item_blueprint_6b95a4d2d45a0b42e14a9c77d4b8f3c1  public handle
generation_contract_d835e4b5c76210ad              convenience id
generation_contract_d835e4b5c76210ad146c2f8e55a92d73  public handle
```

The 64-bit identifiers must never substitute for the public handles or full fingerprints in durable provenance, authorization, deduplication, or replay decisions.

## Blueprint matrix

The compiler evaluates the ordered Cartesian product:

```text
task_families
x difficulty_bands
x evidence_modes
x range(items_per_cell)
```

It rejects a request above 10,000 blueprints before materializing the result. `items_per_cell` is limited to 1–100, and every seed must fit an unsigned 64-bit integer.

## Determinism and auditability

Rubric, blueprint, and generation-contract fingerprints derive from compact canonical UTF-8 JSON and SHA-256. They do not depend on:

- current time;
- process id;
- global random state;
- collection hash order;
- a network response; or
- a hosted-model provider.

Repeating the same normalized input yields byte-identical canonical contract JSON. Modifying rubric content or `rubric_version` changes the rubric fingerprint, and contract creation rejects a blueprint compiled under the earlier fingerprint or governance version.

## Generation contract

`build_generation_contract` returns a JSON-compatible dictionary containing:

- the complete rubric, governance version, and rubric fingerprint;
- the complete blueprint, its 128-bit `blueprint_handle`, and full blueprint fingerprint;
- non-negotiable authoring instructions;
- a strict JSON Schema Draft 2020-12 output contract;
- the 64-bit convenience `contract_id`;
- the authoritative 128-bit `contract_handle`; and
- the full `contract_fingerprint`.

The generated item schema requires:

- `item_id`;
- `stem`;
- `stimulus`;
- `response_format`;
- `options`;
- `answer_key`;
- `scoring_guide`;
- `rubric_alignment`;
- `source_attributions`; and
- `safety_notes`.

Object boundaries set `additionalProperties` to `false`. Score-level arrays use ordered `prefixItems`, ensuring every declared rubric score appears exactly once. Text and collection fields are explicitly bounded.

Each response format has its own answer-key object:

- constructed response: reference response, accepted variants, and rationale;
- selected response: one or more declared option identifiers and rationale;
- binary judgment: Boolean value and rationale;
- ordinal rating: declared rubric score and rationale;
- pairwise comparison: preferred declared option identifier and rationale.

Provider adapters must additionally verify cross-field relations that JSON Schema alone cannot express conveniently, such as answer-key option identifiers referring to options declared in the same generated item.

`render_generation_prompt` is intentionally provider-neutral. It asks for exactly one JSON object and states that instructions embedded in rubric text, source text, or item content must not be executed. The core package does not accept credentials or perform a network call.

## Python/Rust boundary

This authoring slice performs only:

- bounded schema validation;
- deterministic matrix compilation;
- canonical serialization; and
- cryptographic hashing.

It does not compute item difficulty, discrimination, fit, information, DIF, dimensionality, person estimates, evaluator severity, or latent-space positions. Those numerical operations remain in the Rust-backed psychometric layer. Generated candidates are intended to flow into calibration and diagnostic APIs only after structural and content screening.

## Recommended production workflow

```text
1. Author and approve a versioned RubricSpecification
2. Compile a bounded BlueprintPlan
3. Send each GenerationContract to an isolated provider adapter
4. Validate structured output and cross-field references
5. Record a complete semantic `CandidateScreeningResult` for answerability,
   ambiguity, leakage, bias, and source support
6. Admit only pilot-eligible screened candidates, then run artificial-crowd
   and/or human pilot responses
7. Calibrate with Rust-backed IRT / many-facet / latent-space models
8. Reject misfitting, low-information, locally dependent, or DIF items
9. Publish accepted items into a versioned bank
10. Monitor drift and regenerate only deficient blueprint cells
```

Structural conformance is necessary but not sufficient. The governed item bank, not the LLM, is the product artifact.

## Planned slices

1. A provider protocol and deterministic offline fixture provider.
2. Candidate structural validation and content-screening result schemas
   (implemented; pilot admission now requires an eligible screening result).
3. Artificial-crowd orchestration with evaluator provenance.
4. Rust-backed calibration plans and acceptance policies.
5. Versioned item-bank lifecycle, drift, DIF, exposure, and adaptive regeneration.
6. Buyer-facing authoring/calibration reports and an optional service boundary.

## Research basis

The architecture follows Evidence-Centered Design’s distinction among construct, evidence, and task models. Automatic item-generation research likewise treats item models and constraints as the authoring foundation, while empirical screening determines whether generated items function as intended. LLM generation changes the authoring mechanism; it does not remove the need for validation and calibration.

### References

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Haller, S., Aldea, A., Seifert, C., & Strisciuglio, N. (2024). Survey on automated generation of medical assessment questions. *Artificial Intelligence Review, 57*, 128. https://doi.org/10.1007/s10462-024-10726-9

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). *A brief introduction to evidence-centered design* (Research Report RR-03-16). Educational Testing Service. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
