# Release supply-chain evidence

This document records the primary-source basis and the exact scope of the release SBOM and provenance flow in `.github/workflows/publish-pypi.yml`. It is evidence for PRD-FR-081; it does not redefine the psychometric, numerical, packaging, or release-version contracts.

## Evidence model

The release workflow uses two different evidence objects and keeps their claims separate.

1. Each sdist and wheel is attested in the job that builds it. The attestation subject is the distribution file in that builder's `dist/` directory. This is build-provenance evidence: it binds the emitted artifact digest to the GitHub Actions build context that produced it.
2. A dedicated job checks out exactly `inputs.release_commit`, scans that source tree with the pinned Syft release, and emits `fast-mlsirm.spdx.json`. The SBOM is therefore a source-release inventory for that reviewed commit. It is uploaded as a distinct GitHub release asset and its own file is build-provenance-attested.
3. Both irreversible publication sinks depend on successful distribution builds and successful SBOM generation. The PyPI job still downloads only `dist-*`, so the SBOM JSON cannot be uploaded as a Python distribution by accident.

The current source-tree SBOM is deliberately **not** described as a per-wheel binary SBOM or as a GitHub SBOM predicate bound to each wheel/sdist. GitHub's current SBOM-attestation guidance uses `actions/attest` with both `subject-path` and `sbom-path` when an SBOM predicate is intended to describe a specific artifact. If a future requirement needs wheel-specific component inventories, that is a separate contract: generate/validate an SBOM for each built distribution and attest that artifact/SBOM relationship rather than silently broadening the claim made by the source-tree SBOM.

Likewise, this workflow does not claim a SLSA Build Level merely because it emits provenance. SLSA levels include requirements beyond the presence of an attestation, including properties of the build platform and verification model. Any SLSA-level claim requires a separate assessment against the then-current approved specification.

## Primary-source basis

- SLSA Community. (2026). *SLSA specification (Version 1.2)*. Linux Foundation. https://slsa.dev/spec/v1.2/  
  Scope used here: provenance is verifiable information connecting artifacts to where, when, and how they were produced. The specification is the authority for the meaning and limits of a SLSA provenance claim; this repository uses that model without asserting an unverified SLSA level.
- SLSA Community. (2026). *Build: Provenance (SLSA Version 1.2)*. Linux Foundation. https://slsa.dev/spec/v1.2/build-provenance/  
  Scope used here: build provenance is artifact-oriented and is intended to let consumers trace build outputs to the source and build process. This supports creating provenance in the same job that produces each sdist/wheel instead of reconstructing builder identity later in an aggregation job.
- SPDX Workgroup. (2022). *Software Package Data Exchange (SPDX) Specification Version 2.3*. Linux Foundation. https://spdx.github.io/spdx-spec/v2.3/  
  Scope used here: SPDX is the interchange format family selected for the standalone release SBOM. The release gate must inspect the emitted document's declared `spdxVersion`; the workflow does not infer a format revision solely from the `spdx-json` option name.
- GitHub. (2026). *Using artifact attestations to establish provenance for builds*. GitHub Docs. https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations  
  Scope used here: GitHub requires `contents: read`, `id-token: write`, and `attestations: write` for artifact attestations; `subject-path` identifies the artifact being attested. The same guidance distinguishes ordinary build provenance from an SBOM attestation, which uses a subject artifact together with an `sbom-path`.

Sources were checked on 2026-09-05. Versioned standards are cited by version so a later website default cannot silently change the historical release basis.

## Release verification

For a candidate release commit, acceptance evidence must establish all of the following on one unchanged integrated protected head:

- the sdist and every wheel were built from the reviewed release commit and their builder-local provenance steps succeeded;
- `fast-mlsirm.spdx.json` was generated from that same `inputs.release_commit`, its generation/provenance step succeeded, and the release asset is byte-identical to the uploaded workflow artifact;
- the emitted SBOM declares the expected SPDX version and parses as JSON; unsupported or malformed output fails the release rather than being relabeled;
- the GitHub release sink receives package distributions plus the separately named SBOM asset;
- the PyPI sink receives only `dist-*` package artifacts and remains gated on the SBOM job without downloading `release-sbom`;
- artifact attestations can be verified against `ContextualWisdomLab/fast-mlsirm` with the GitHub CLI before release evidence is accepted;
- no SLSA level, per-wheel SBOM coverage, Trusted Publisher configuration, or PyPI-native attestation claim is made unless a separate exact-head contract verifies it.

This separation keeps the evidence auditable: build provenance answers how a distribution was produced; the standalone SPDX document inventories the reviewed release source; publication dependencies prove that neither sink can complete while required evidence generation has failed.
