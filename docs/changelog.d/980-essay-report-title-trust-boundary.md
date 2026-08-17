# Essay report title trust boundary

## Fixed

- Hardened `render_essay_score_report_html()` so caller-supplied titles admit only exact built-in strings, rejecting caller-controlled `str` subclasses before overridden text callbacks such as `strip()` or HTML-escaping operations can execute.
- Added a hostile-string-subclass regression that proves rejection occurs before callback execution or artifact creation; scoring, calibration, and psychometric arithmetic remain unchanged.
