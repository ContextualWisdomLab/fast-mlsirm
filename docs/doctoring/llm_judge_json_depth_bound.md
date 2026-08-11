# LLM judge JSON nesting depth bound

## Standard / threat model

Open Web Application Security Project. (2021). *OWASP API security top 10 2023: API4 — unrestricted resource consumption*. OWASP Foundation. https://owasp.org/API-Security/

Bray, T. (Ed.). (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). Internet Engineering Task Force. https://doi.org/10.17487/RFC8259

## Product application

Automated LLM-judge responses are untrusted text. Before `json.loads`, the product walks the raw response character stream, tracks object/array nesting outside string literals, and rejects nesting deeper than 32 with `JudgeFormatError`. This bounds parser stack/heap growth for recursive structures while preserving legitimate shallow score objects used by Contextual Orchestrator judges.

## Verification

- `tests/test_llm_judge.py::test_judge_rejects_excessive_json_nesting`
- `tests/test_llm_judge.py::test_judge_accepts_bounded_json_nesting`
