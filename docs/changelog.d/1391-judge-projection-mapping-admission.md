# Judge projection mapping admission

## Fixed

- LLM-as-a-Judge construct projection now requires exact built-in criterion-score and criterion-category dictionaries before any mapping iteration or lookup. Caller-defined mapping protocols therefore cannot synthesize or replace criterion evidence during the IRT handoff, while exact `dict` inputs preserve the existing explicit item-order and category semantics.
