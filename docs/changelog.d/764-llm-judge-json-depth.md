# LLM judge JSON nesting depth bound

## Fixed

- Cap raw LLM-judge response JSON nesting at 32 levels before `json.loads`, failing closed with `JudgeFormatError` so hostile recursive objects cannot expand into parser resource exhaustion.
- Keep valid shallow judge payloads accepted with the existing criterion/score contracts.
