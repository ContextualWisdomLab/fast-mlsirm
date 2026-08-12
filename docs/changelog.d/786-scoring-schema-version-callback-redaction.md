# Scoring schema-version callback redaction

## Fixed

- Assessment schema-version validation now requires an exact built-in `str` matching the wire version, rejecting hostile string subclasses before equality work so callback messages cannot leak into contract errors.
