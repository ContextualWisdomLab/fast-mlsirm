# Essay report title trust boundary

## Fixed

- Hardened score, validation-evidence, and facets-calibration essay HTML renderers so caller-supplied titles admit only exact built-in strings, rejecting caller-controlled `str` subclasses before overridden text callbacks such as `strip()` or HTML-escaping operations can execute.
- Added hostile-string-subclass regressions that prove all three public renderers reject before callback execution or artifact creation; scoring, calibration estimation, and psychometric arithmetic remain unchanged.
