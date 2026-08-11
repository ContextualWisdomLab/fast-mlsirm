# Governed Item-Bank Lifecycle Contract

## Scope

This record governs the first post-pilot item-bank lifecycle slice in `fast-mlsirm`. It defines immutable state transitions and source-text-free evidence references. It does **not** estimate item parameters, establish validity, approve an item automatically, create a hosted workflow, or define a physical database schema.

The existing rubric, generated-candidate audit, and verified pilot-admission contracts remain authoritative before this lifecycle begins:

```text
GeneratedItemCandidate
  -> CandidateAuditReport
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

The referenced artifact may contain Rust-backed calibration, fit, DIF, information, linking, exposure, drift, approval, suspension, or retirement evidence. The lifecycle layer does not reproduce or reinterpret those calculations.

The initial calibration gate always requires calibration, item-fit, and item-information evidence plus **exactly one DIF-applicability branch**:

- `dif` when the governed calibration design includes a comparison/grouping structure for which DIF evidence is applicable; or
- `dif_not_applicable` when a separate governed evidence artifact establishes that the calibration design has no applicable DIF comparison.

Supplying neither branch fails closed; supplying both fails closed as conflicting applicability evidence. `dif_not_applicable` is not a fairness, invariance, validity, or cross-population equivalence finding. It only prevents callers from fabricating a DIF result for a design in which DIF is scientifically inapplicable. Any approved use, population, language, domain, release, or score-comparability claim that requires DIF/invariance evidence must still provide the applicable evidence before that claim is made.

These requirements prove only that the governed evidence classes are present; later slices must validate their schemas, estimator identity, uncertainty, population/design scope, parameter recovery, and decision thresholds.

Approval is use-specific. An item cannot enter `approved` without at least one descriptive approved-use identifier, and an `active` record cannot erase that scope.

## Psychometric discrimination versus policy criticality

`PolicyCriticality` is explicitly separate from discrimination, information, fit, or reliability.

- `ordinary` means no additional operational criticality is asserted by this contract.
- `required` means downstream assembly policy requires the item or criterion for the approved use.
- `conjunctive_gate` means failure cannot be offset by a higher aggregate score.

A low-information safety-critical criterion may therefore remain operationally required. Conversely, a highly discriminating item is not automatically safe, fair, valid, or approved.

## Suspension and retirement

Suspension is reversible but requires both a governance suspension record and evidence of a measured concern such as drift or DIF. A `dif_not_applicable` calibration determination is not evidence of a later operational concern and therefore does not satisfy the suspension gate. Reactivation requires new approval plus new drift evidence. Retirement is terminal for new operational use, while historical evidence remains content-addressed for audit reconstruction.

The contract does not authorize physical deletion, retention exceptions, or erasure behavior. Those controls belong to the downstream persistence and governance system.

## Security and privacy boundary

Canonical lifecycle records contain no source, response, prompt, provider-output, or rejected-content text. Complete content fingerprints support exact provenance but are not signatures, identities, anonymity guarantees, authentication credentials, or authorization decisions.

Downstream systems remain responsible for purpose-bound access, tenant isolation, encryption, key management, retention/export/deletion, legal basis, human approval identity, and tamper-evident audit storage.

Provider/model output cannot create calibration, approval, activation, suspension, retirement, repository, merge, release, or deployment authority by identity alone.

## Verification requirements

The implementation requires deterministic tests for:

- exact verified-pilot provenance;
- direct-construction refusal;
- transition graph enforcement;
- required evidence classes;
- mutually exclusive measured-DIF versus governed-DIF-not-applicable calibration evidence;
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

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to evidence-centered design. *ETS Research Report Series, 2003*(1), i–29. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
