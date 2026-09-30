# Keep malformed JSON depth budgets monotonic

## Fixed

Status: **Proposed in PR #2254; not protected-main authority**

- The proposed repair prevents unmatched closing delimiters from lowering the
  raw JSON depth counter below zero in rubric-provider or LLM-judge response
  parsing. Malformed input would fail at the bounded-depth guard consistently
  before decoder syntax validation; the decoder's strict rejection behavior
  remains unchanged.
- Proposed regression evidence covers both affected untrusted string parser
  boundaries. The change removes an unrelated test of the already-correct
  bounded-file helper.
