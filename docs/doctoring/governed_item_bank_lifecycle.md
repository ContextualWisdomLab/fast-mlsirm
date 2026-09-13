# Governed Item-Bank Lifecycle Contract

## Scope

This record governs the first post-pilot item-bank lifecycle slice in `fast-mlsirm`. It defines immutable state transitions and source-text-free evidence references. It does **not** estimate item parameters, establish validity, approve an item automatically, create a hosted workflow, or define a physical database schema.

The existing rubric, generated-candidate audit, and verified pilot-admission contracts remain authoritative before this lifecycle begins:

```text
GeneratedItemCandidate
  -> CandidateAuditReport
  -> CandidateScreeningResult
  -> PilotCandidateRecord
  -> ItemBankLifecycleRecord(piloting)
```

The lifecycle then permits only:

```text
piloting -> calibrated -> approved -> active
active -> suspended -> active
active -> retired
suspended -> retired
```

Every transition creates a new content-addressed record linked to the exact previous record fingerprint. No operational record is edited in place.

## Evidence and interpretation boundary

A lifecycle evidence reference stores only:

- an evidence kind;
- a descriptive evidence identifier; and
- the complete SHA-256 fingerprint of the exact evidence artifact.

The referenced artifact may contain Rust-backed calibration, fit, DIF, information, linking, exposure, or drift evidence, as well as source-text-free evidence/content-validity, security/privacy, approval, suspension, or retirement governance evidence. The lifecycle layer does not reproduce or reinterpret numerical calculations or infer that the presence of a reference proves the underlying concern is resolved.

The initial calibration gate requires references for calibration, item fit, DIF, and item information. This requirement proves only that the governed evidence classes are present; later slices must validate their schemas, estimator identity, uncertainty, population/design scope, parameter recovery, and decision thresholds.

Approval is use-specific. An item cannot enter `approved` without at least one descriptive approved-use identifier, and an `active` record cannot erase that scope.

## Psychometric discrimination versus policy criticality

`PolicyCriticality` is explicitly separate from discrimination, information, fit, or reliability.

- `ordinary` means no additional operational criticality is asserted by this contract.
- `required` means downstream assembly policy requires the item or criterion for the approved use.
- `conjunctive_gate` means failure cannot be offset by a higher aggregate score.

A low-information safety-critical criterion may therefore remain operationally required. Conversely, a highly discriminating item is not automatically safe, fair, valid, or approved.

## Suspension and retirement

Suspension is reversible but requires both a governance suspension reference and newly supplied concern evidence. Governed concern classes include DIF, drift, exposure, linking, evidence validity, content validity, and security/privacy, so a non-psychometric quarantine does not need fabricated DIF or drift evidence. The suspended record binds the exact newly asserted concern classes in its content-addressed identity.

Reactivation requires a newly supplied approval plus newly supplied evidence for every concern class bound by that suspension; unrelated evidence cannot clear a different quarantine (for example, a fresh DIF artifact cannot reactivate an item suspended for security/privacy). “Newly supplied” means a content fingerprint not already present anywhere in the suspended record’s cumulative evidence history. A caller cannot make historical approval or concern evidence appear fresh by assigning it a new `evidence_id`; approval and every persisted suspension concern must bind genuinely new evidence content.

These gates establish evidence presence and provenance only; downstream governance remains responsible for deciding whether the referenced evidence actually resolves the operational concern. Retirement is terminal for new operational use, while historical evidence remains content-addressed for audit reconstruction.

The contract does not authorize physical deletion, retention exceptions, or erasure behavior. Those controls belong to the downstream persistence and governance system.

## Security and privacy boundary

Canonical lifecycle records contain no source, response, prompt, provider-output, or rejected-content text. Complete content fingerprints support exact provenance but are not signatures, identities, anonymity guarantees, authentication credentials, or authorization decisions.

Downstream systems remain responsible for purpose-bound access, tenant isolation, encryption, key management, retention/export/deletion, legal basis, human approval identity, and tamper-evident audit storage.

Provider/model output cannot create calibration, approval, activation, suspension, retirement, repository, merge, release, or deployment authority by identity alone.

## Cross-version comparability (item-bank release)

The `ItemBankEntry` contract (`python/fast_mlsirm/scoring/item_bank.py`) accumulates lifecycle evidence for a single item version through immutable state transitions. An `ItemBankRelease` bundles exact entry fingerprints into a versioned manifest and declares whether scores across successive releases can be meaningfully compared.

Cross-version comparability is a psychometric claim, not a technical default. When `ItemBankRelease.cross_version_comparable` is `True`, the contract requires both:

- a `predecessor_release_fingerprint` identifying the exact prior release; and
- non-empty `linking_evidence_fingerprints` referencing IRT scale-linking or equating evidence (e.g., common-item Haebara or Stocking-Lord coefficients, moment-method transformations, observed-score concordance tables).

These requirements enforce that comparability claims rest on verifiable psychometric evidence rather than assertion alone. A release without linking evidence or predecessor provenance may still be valid for operational use but must not claim cross-version score comparability.

## Verification requirements

The implementation requires deterministic tests for:

- exact verified-pilot provenance;
- direct-construction refusal;
- transition graph enforcement;
- required evidence classes;
- non-psychometric suspension and reactivation without fabricated DIF/drift evidence;
- suspension-to-reactivation concern-class continuity;
- rejection of historical approval or concern fingerprints under replacement identifiers;
- use-specific approval;
- cumulative evidence and previous-record linkage;
- evidence-order invariance;
- conflicting evidence identifiers;
- policy-criticality preservation;
- post-construction mutation replay failure;
- exact package-owned child types;
- terminal retirement; and
- redacted stable errors.

No release claim follows until the unchanged integrated head passes full repository CI, security, coverage, packaging, review, and release-acceptance gates.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Haebara, T. (1980). Equating logistic ability scales by a weighted least squares method. *Japanese Psychological Research, 22*(3), 144–149. https://doi.org/10.4992/psycholres1954.22.144

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking: Methods and practices* (3rd ed.). Springer. https://doi.org/10.1007/978-1-4939-0317-7

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to evidence-centered design. *ETS Research Report Series, 2003*(1), i–29. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x

Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item response theory. *Applied Psychological Measurement, 7*(2), 201–210. https://doi.org/10.1177/014662168300700208

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
