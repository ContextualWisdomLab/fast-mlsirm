# Harden generic diagnostics report title callback boundary

## Fixed

- Reject caller-defined `str` subclasses at the public generic diagnostics-report title boundary before truth-value or HTML-escaping callbacks can run, while preserving `None` and an empty exact built-in string as requests for the report-type default title.
