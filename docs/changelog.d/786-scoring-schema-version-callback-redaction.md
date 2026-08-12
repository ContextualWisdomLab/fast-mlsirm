# Scoring schema-version callback redaction

## Fixed

- Assessment schema-version validation admits only exact built-in strings so
  hostile equality callbacks cannot execute during evidence reference
  construction.
