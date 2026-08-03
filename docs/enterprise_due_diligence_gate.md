# Enterprise Due-Diligence Gate

## Purpose

`enterprise_due_diligence_gate` is the canonical name for the repository's commercial evidence gate. The name is intentionally amount-neutral: it describes the evidence required for enterprise procurement review, not a company valuation, contract guarantee, or regulated-use approval.

The first migration slice is implemented by:

```bash
python scripts/enterprise_due_diligence_gate.py \
  --source-commit "$(git rev-parse HEAD)" \
  --require-enterprise-due-diligence \
  --currency-code KRW \
  --scenario-amount 2000000000 \
  --out enterprise_due_diligence_gate.json
```

## Three distinct concepts

The repository must keep these concepts separate in code, reports, filenames, and buyer communication:

1. **Software evidence gate** — `enterprise_due_diligence_gate`; an amount-neutral contract that verifies evidence completeness and provenance.
2. **Procurement scenario** — a currency-explicit amount such as KRW 2,000,000,000, represented by `currency_code` and `scenario_amount`.
3. **Enterprise-value thesis** — an aspirational USD 20,000,000,000 product ambition that requires independently verified market adoption, revenue, retention, defensibility, and customer value. It is not established by a repository test result.

## Manifest contract

The deterministic JSON manifest includes:

- `gate_name`: always `enterprise_due_diligence_gate`;
- `currency_code`: a normalized three-letter ASCII code;
- `scenario_amount`: a positive integer in the declared currency;
- `scenario_name`: a currency- and amount-explicit procurement identifier;
- `valuation_claim`: always `false`;
- `source_commit`: the immutable source identifier represented by the evidence;
- `schema_version`: the gate-contract version;
- `legacy_gate_aliases`: bounded aliases accepted only during migration.

No timestamp is included in this manifest so identical inputs produce byte-stable JSON.

## Compatibility and deprecation

The utility temporarily accepts the legacy aliases `20b`, `20b_product`, `20b_product_readiness`, and `require_20b_product`. It also accepts the hidden CLI flag `--require-20b-product`. Every legacy path emits a `DeprecationWarning` and writes only the canonical gate name.

New automation must use:

```text
--require-enterprise-due-diligence
```

The legacy compatibility path must be removed only after active workflows, scripts, report contracts, examples, and documentation have migrated and the removal is recorded in the changelog.

## Safety boundary

A successful gate means that declared evidence met the implemented contract for the stated source commit and procurement scenario. It does not mean that:

- a sale or acquisition will occur;
- the software is worth the scenario amount;
- the product has achieved a USD 20 billion valuation;
- high-stakes deployment is valid without domain-specific validation and governance;
- external security, legal, privacy, accessibility, or scientific review can be skipped.

## Follow-up integration

Subsequent reviewable slices should migrate `scripts/sales_readiness.py`, `scripts/build_commercial_release.py`, generated JSON/HTML contracts, workflow flags, buyer-facing documentation, and active filenames. Historical plan/spec files may retain their original names only when an audit header states the original KRW 2,000,000,000 scenario and explicitly disclaims valuation meaning.
