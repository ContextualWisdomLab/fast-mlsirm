# Governed automated-essay scoring adapters

## Added

- Added `fast_mlsirm.scoring.essay` domain adapters that bind exact prompt, submission, evidence-span, assessment, rubric, task-family, criterion, and engine provenance to the existing provider-neutral scoring contracts without storing raw prompt, essay, or source text.
- Added factory-sealed, content-addressed prompt, submission, evidence, and request artifacts; deterministic review flags; bounded source-text-free offsets; replay and source-identity checks; shared engine-policy authorization; and deterministic human/automated fixture coverage.
- The adapter introduces no parallel rubric, observation, result, or provider schema and makes no claim of scoring accuracy, reliability, fairness, scoreability, construct validity, or readiness for consequential automation.
