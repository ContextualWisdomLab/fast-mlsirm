# Governed item-bank lifecycle reporting

## Decision

`fast-mlsirm` exposes a report layer over the existing immutable item-bank lifecycle records. The report accepts a complete contiguous lineage beginning at the package-created `piloting` record and emits deterministic machine-readable JSON-compatible content or a standalone accessible HTML representation.

The report is deliberately provenance-only. It may state that calibration, item-fit, DIF, item-information, linking, exposure, drift, approval, suspension, or retirement evidence is present because the lifecycle record binds an exact evidence identifier and SHA-256 fingerprint. It does not recompute, reinterpret, or synthesize those numerical results in Python. Production calibration, fit, DIF, information, linking/equating, exposure, drift, and uncertainty arithmetic remain Rust-owned.

## Buyer-facing truth conditions

- The current lifecycle state is taken from the final record in a complete fingerprint-linked lineage; a disconnected successor cannot be rendered as an authoritative timeline.
- Approved-use scope, rubric/blueprint identity, candidate/pilot/audit fingerprints, policy criticality, evidence identities, transition reasons, and record fingerprints remain visible without embedding raw source, prompt, response, or provider content.
- Cross-version comparability is reported as `supported_by_linking_evidence` only when the governed evidence inventory contains a linking reference. Otherwise it is explicitly `not_demonstrated`; nominally equal score ranges and an `active` state are insufficient.
- Missing linking, exposure, and drift evidence are rendered as explicit limitations rather than silently omitted. A piloting-only report also states that calibration is not complete.
- The HTML representation is standalone, uses semantic headings/tables and captions, includes a skip link, escapes the caller-supplied title, and preserves a visible `:focus-visible` keyboard indicator.

## Security and privacy boundary

The report takes exact package-owned `ItemBankLifecycleRecord` values in a built-in tuple. It rejects container subclasses before iteration and rejects title subclasses before string callbacks. Report payloads contain governed metadata and fingerprints only; fingerprints are provenance identifiers, not authorization, anonymization, or signatures. Downstream persistence, tenant authorization, retention, encryption, export/deletion, and audit policy remain owned by the downstream application such as Psychometrics Commons.

## Research and standards basis

The lifecycle report follows evidence-centered assessment design by keeping claims connected to explicit evidence rather than treating generated content as measurement truth. Automatic item generation research also motivates retaining item-model/bank provenance instead of treating generated items as interchangeable. The current published *Standards for Educational and Psychological Testing* remains the 2014 joint AERA/APA/NCME edition while a revision process is underway; this implementation therefore cites the final 2014 standard rather than a draft revision.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association. https://www.testingstandards.net/open-access-files.html

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). *A brief introduction to evidence-centered design* (Research Report RR-03-16). Educational Testing Service. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
