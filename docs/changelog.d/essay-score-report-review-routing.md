# Provenance-bound essay score reports

## Added

- Added a provider-neutral, content-addressed `EssayScoreReport` adapter over the existing governed essay request, shared scoring result, and engine descriptor.
- Report construction replays exact request, engine, assessment, rubric, construct, granularity, and criterion provenance before emission.
- Submission review flags, terminal observations, and scored observations without evidence now produce non-suppressible transparent human-review triggers. The report preserves criteria separately and explicitly does not treat absence of a trigger as validity or deployment evidence.
