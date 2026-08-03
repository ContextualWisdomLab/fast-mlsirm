# Currency-explicit enterprise due-diligence gate

## Added

- Added a deterministic `enterprise_due_diligence_gate` manifest utility that separates the amount-neutral software evidence gate from a currency-explicit procurement scenario and always records `valuation_claim: false`.
- Added a bounded deprecation bridge for legacy `20b` gate aliases and `--require-20b-product`, with canonical output and explicit warnings.
- Documented the distinction among enterprise evidence, the KRW 2,000,000,000 procurement scenario, and the aspirational USD 20,000,000,000 enterprise-value thesis.
