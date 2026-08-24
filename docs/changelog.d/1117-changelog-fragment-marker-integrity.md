# Changelog fragment marker integrity

## Fixed

- Reject authoritative changelog fragments containing reserved managed-block marker literals before rendering or update, preventing nested markers from producing a changelog that fails its own next integrity check.
