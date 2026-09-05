# Release supply-chain evidence

This document records the primary-source basis and the exact scope of the release SBOM and provenance flow in `.github/workflows/publish-pypi.yml`. It is evidence for PRD-FR-081; it does not redefine the psychometric, numerical, packaging, or release-version contracts.

## Evidence model

The release workflow uses two different evidence objects and keeps their claims separate.

1. Each sdist and wheel is attested in the job that builds it. The attestation subject is the distribution file in that builder's `dist/` directory. This is build-provenance evidence: it binds the emitted artifact digest to the GitHub Actions build context that produced it.
2. A dedicated job checks out exactly `inputs.release_commit`, scans that source tree with the pinned Syft release, and emits `fast-mlsirm.spdx.json`. The SBOM is therefore a source-release inventory for that reviewed commit. It is uploaded as a distinct GitHub release asset and its own file is build-provenance-attested.
3. Before attestation or publication, repository-owned `scripts/validate_release_spdx.py` parses that artifact as strict interoperable JSON and replays the mandatory SPDX 2.3 document-creation envelope: `spdxVersion == "SPDX-2.3"`, `dataLicense == "CC0-1.0"`, `SPDXID == "SPDXRef-DOCUMENT"`, a non-blank document `name`, an absolute fragment-free `documentNamespace` whose percent escapes satisfy RFC 3986, and `creationInfo` containing a UTC `created` timestamp plus at least one non-blank creator. A format option or version string alone is not accepted as proof that the emitted file is a valid SPDX 2.3 document envelope.
4. Both irreversible publication sinks depend on successful distribution builds and successful SBOM generation/validation/attestation. The PyPI job still downloads only `dist-*`, so the SBOM JSON cannot be uploaded as a Python distribution by accident.

The current source-tree SBOM is deliberately **not** described as a per-wheel binary SBOM or as a GitHub SBOM predicate bound to each wheel/sdist. GitHub's current SBOM-attestation guidance uses `actions/attest` with both `subject-path` and `sbom-path` when an SBOM predicate is intended to describe a specific artifact. If a future requirement needs wheel-specific component inventories, that is a separate contract: generate/validate an SBOM for each built distribution and attest that artifact/SBOM relationship rather than silently broadening the claim made by the source-tree SBOM.

Likewise, this workflow does not claim a SLSA Build Level merely because it emits provenance. SLSA levels include requirements beyond the presence of an attestation, including properties of the build platform and verification model. Any SLSA-level claim requires a separate assessment against the then-current approved specification.

## Primary-source basis

- SLSA Community. (2026). *SLSA specification (Version 1.2)*. Linux Foundation. https://slsa.dev/spec/v1.2/  
  Scope used here: provenance is verifiable information connecting artifacts to where, when, and how they were produced. The specification is the authority for the meaning and limits of a SLSA provenance claim; this repository uses that model without asserting an unverified SLSA level.
- SLSA Community. (2026). *Build: Provenance (SLSA Version 1.2)*. Linux Foundation. https://slsa.dev/spec/v1.2/build-provenance/  
  Scope used here: build provenance is artifact-oriented and is intended to let consumers trace build outputs to the source and build process. This supports creating provenance in the same job that produces each sdist/wheel instead of reconstructing builder identity later in an aggregation job.
- SPDX Workgroup. (2022). *Software Package Data Exchange (SPDX) Specification Version 2.3*. Linux Foundation. https://spdx.github.io/spdx-spec/v2.3/  
  Scope used here: SPDX is the interchange format family selected for the standalone release SBOM. SPDX 2.3 requires the document-creation identity and provenance fields represented in JSON as `spdxVersion`, `dataLicense`, `SPDXID`, `name`, `documentNamespace`, and `creationInfo` with creator(s) and UTC creation time. Clause 6.5 further requires the document namespace to be a unique absolute URI and forbids the `#` fragment delimiter because SPDX element URIs append their own fragment identifiers. The release gate validates those fields in the emitted artifact rather than inferring conformance solely from the generator's `spdx-json` option name.
- Berners-Lee, T., Fielding, R., & Masinter, L. (2005). *Uniform Resource Identifier (URI): Generic Syntax (RFC 3986)*. Internet Engineering Task Force. https://www.rfc-editor.org/rfc/rfc3986  
  Scope used here: the SPDX document namespace is an absolute URI, so its percent-encoded octets must follow RFC 3986 section 2.1 (`%` followed by exactly two hexadecimal digits). Python's structural URL splitter does not itself prove this lexical invariant; malformed percent escapes therefore fail closed before the namespace is accepted as release evidence.
- SPDX Workgroup. (2023). *SPDX and NTIA Minimum Elements for SBOM HOWTO*. Linux Foundation. https://spdx.github.io/spdx-ntia-sbom-howto/  
  Scope used here: the SPDX 2.x document-creation section's mandatory fields are SPDX version, data license, SPDX identifier, document name, document namespace, creator, and created timestamp. This is the executable minimum envelope replayed by `validate_release_spdx.py`; it is not presented as full semantic validation of every optional SPDX relation or package field.
- Bray, T. (2017). *The JavaScript Object Notation (JSON) Data Interchange Format (RFC 8259)*. Internet Engineering Task Force. https://www.rfc-editor.org/rfc/rfc8259  
  Scope used here: JSON object member names should be unique for interoperable interpretation; receivers do not agree reliably on duplicate-member behavior. Release SBOM validation therefore rejects duplicate members at every object depth as well as non-standard numeric constants before provenance attestation or publication.
- GitHub. (2026). *Using artifact attestations to establish provenance for builds*. GitHub Docs. https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations  
  Scope used here: GitHub requires `contents: read`, `id-token: write`, and `attestations: write` for artifact attestations; `subject-path` identifies the artifact being attested. The same guidance distinguishes ordinary build provenance from an SBOM attestation, which uses a subject artifact together with an `sbom-path`.

Sources were checked on 2026-09-05. Versioned standards are cited by version so a later website default cannot silently change the historical release basis.

## Release verification

For a candidate release commit, acceptance evidence must establish all of the following on one unchanged integrated protected head:

- the sdist and every wheel were built from the reviewed release commit and their builder-local provenance steps succeeded;
- `fast-mlsirm.spdx.json` was generated from that same `inputs.release_commit`, its validation/generation/provenance steps succeeded, and the release asset is byte-identical to the uploaded workflow artifact;
- the emitted SBOM parses as interoperable JSON; malformed syntax, non-standard numeric constants, and duplicate object members fail closed rather than being normalized or collapsed;
- the document carries the required SPDX 2.3 creation envelope: exact SPDX/data-license/document identifiers, non-blank document name, absolute fragment-free namespace URI with syntactically valid RFC 3986 percent escapes, UTC creation timestamp, and at least one non-blank creator; missing, malformed, or unsupported values fail before attestation/publication;
- the GitHub release sink receives package distributions plus the separately named SBOM asset;
- the PyPI sink receives only `dist-*` package artifacts and remains gated on the SBOM job without downloading `release-sbom`;
- artifact attestations can be verified against `ContextualWisdomLab/fast-mlsirm` with the GitHub CLI before release evidence is accepted;
- no SLSA level, per-wheel SBOM coverage, Trusted Publisher configuration, or PyPI-native attestation claim is made unless a separate exact-head contract verifies it.

This separation keeps the evidence auditable: build provenance answers how a distribution was produced; the standalone SPDX document inventories the reviewed release source; publication dependencies prove that neither sink can complete while required evidence generation has failed.
