# PR queue bounded JSON import

## Fixed

- Remove the plain-`json.loads()` isolation fallback from PR queue snapshot capture so queue-governance evidence always uses the repository's bounded JSON parser or fails closed when that parser is unavailable.
- Distinguish a missing package-layout import from a `ModuleNotFoundError` raised inside the real bounded parser, preserving the sibling direct-script import path without masking broken parser dependencies.

Existing GitHub retry deadlines, capture budgets, malformed-list evidence rejection, duplicate/non-finite/depth JSON policy, and queue identity limits remain unchanged.
