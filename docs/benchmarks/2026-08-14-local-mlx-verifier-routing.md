# Local MLX verifier routing calibration — 2026-08-14

Status: exploratory model-eligibility evidence; not a quality, fairness, or
production IRT claim.

All calls used:

`ContextualOrchestratorJudge -> contextual-orchestrator._FastMLSIJudgeAdapter -> TaskOrchestrator -> ModelClient -> mlx-lm`

The probe used two criteria, K=`3`, implicit `binary_threshold`, disabled
thinking, temperature `0`, bounded output, and zero local retries. A valid
result had two polytomous item columns. Boundary parse and monotonicity failures
were retained as failed comparisons; no keyword matching, positional inference,
retry, category synthesis, or silent repair was used.

| model | safe case | unsafe case | latency | provider tokens |
| --- | --- | --- | ---:| ---:|
| Llama 3B | passed, score `1.0`, `[2,2]` | failed closed, `non_monotone`, `4/4` parsed calls | `6.82 s` / `2.56 s` | `1,855` / `1,854` |
| Gemma 4 e4b | passed, score `1.0`, `[2,2]` | passed, score `0.5`, `[1,1]` | `7.73 s` / `4.87 s` | `1,836` / `1,828` |
| Gemma 4 31B | failed closed, boundary failure, `1/4` calls completed | passed, score `0.25`, `[1,0]` | `96.93 s` / `52.38 s` | `481` / `1,871` |
| DeepSeek R1 Qwen 32B | failed closed, `0/4` calls completed | failed closed, `0/4` calls completed | `100.04 s` / `100.06 s` | `0` / `0` |

This run supports e4b as the current verifier candidate for this workload on
structured-output reliability and bounded latency. It does not establish
unbiasedness, general model superiority, or positive-choice-count bias removal.
The 3B remains a lower-priority candidate; larger models remain discoverable
for non-verifier roles but require a fresh readiness/calibration result before
verifier use.
