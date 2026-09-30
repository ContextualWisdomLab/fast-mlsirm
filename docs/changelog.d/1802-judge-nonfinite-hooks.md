# Reject non-finite judge scores at JSON parse time

## Fixed

- Harden the LLM-as-a-Judge response parser with `parse_constant` and `parse_float` hooks so `NaN`/`Infinity` literals and overflow forms like `1e999` fail closed with an explicit non-finite/unsupported-constant `JudgeFormatError` instead of flowing into score validation with a generic message. No valid finite-score payload, rubric validation, or scoring arithmetic changes.
