# Keep malformed JSON depth budgets monotonic

## Fixed

- Unmatched closing delimiters can no longer lower the raw JSON depth counter
  below zero in rubric-provider or LLM-judge response parsing. Malformed input
  now fails at the bounded-depth guard consistently before decoder syntax
  validation; the decoder's strict rejection behavior is unchanged.
- Regression evidence now covers both affected untrusted string parser
  boundaries. An unrelated test of the already-correct bounded-file helper was
  removed from the change.
