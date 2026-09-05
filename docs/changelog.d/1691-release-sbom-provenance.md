# Release SBOM and builder-local provenance

## Fixed

- Release evidence now requires builder-local provenance for the sdist and every wheel plus a separately published SPDX JSON source-release SBOM generated from the exact reviewed release commit. GitHub release and PyPI publication are both gated on successful SBOM generation, while the PyPI upload set remains restricted to package distributions. The evidence scope and primary-source basis are documented explicitly; the standalone source-tree SBOM is not misrepresented as a per-wheel SBOM predicate or as proof of a SLSA Build Level. The newly introduced SBOM job also uses the repository's explicit Ubuntu 24.04 Linux runner contract rather than reintroducing the moving `ubuntu-latest` alias owned by the separate runner-pinning lane.
