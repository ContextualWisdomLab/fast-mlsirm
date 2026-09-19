# Essay integer admission survives implementation reloads

## Fixed

- Normalize callback-sensitive prompt, submission, and evidence integer inputs to package-trusted exact integers before public factories enter reloadable implementation code, so sequential or re-entrant implementation reloads cannot restore caller-controlled `__index__` execution at the supported package boundary.
- Preserve concrete package-supported NumPy integer compatibility, prompt/submission-specific bounds, and the existing bounded `AssessmentSpecError` contracts without changing essay scoring, calibration, estimation, likelihood, uncertainty, or other psychometric arithmetic.
