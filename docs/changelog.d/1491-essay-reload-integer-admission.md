# Essay integer admission survives implementation reloads

## Fixed

- Re-establish the callback-free essay integer guard at each public prompt, submission, and evidence factory call so reloading the implementation module cannot restore caller-controlled `__index__` execution at the supported package boundary.
- Preserve concrete package-supported NumPy integer compatibility and the existing bounded `AssessmentSpecError` contracts without changing essay scoring, calibration, estimation, likelihood, uncertainty, or other psychometric arithmetic.
