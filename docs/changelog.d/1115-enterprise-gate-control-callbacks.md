# Enterprise gate semantic-control callback safety

## Fixed

- Reject caller-defined string subclasses for enterprise gate names and currency codes before normalization can invoke caller text callbacks.
- Reject caller-defined integer subclasses for procurement scenario amounts before comparison while preserving the positive-integer validation contract for exact built-in values.
