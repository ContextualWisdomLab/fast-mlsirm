# fast-mlsirm

fast-mlsirm is a reusable psychometric measurement toolkit centered on multidimensional latent-space item-response modeling, simulation, fitting, recovery diagnostics, calibration, and evidence-bound evaluation workflows.

## Start here

Use fast-mlsirm when measurement or evaluation work needs explicit model assumptions, reproducible estimation, calibration evidence, and a production numerical path that keeps performance-critical mathematics in the Rust core while exposing a Python package surface.

The repository includes MLSIRM/MLS2PLM work, item-response and calibration utilities, model/evaluation diagnostics, sampling and statistical evidence contracts, and integration surfaces used by other ContextualWisdomLab products.

## Documentation

The canonical documentation map is maintained in [`docs/README.md`](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/README.md). This Pages landing does not duplicate that authority map.

- [Repository overview](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/README.md)
- [Architecture](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/ARCHITECTURE.md)
- [Product/technical gap baseline](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/product-technical-gap-baseline.md)
- [Architecture decisions](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/adr/README.md)
- [Research-to-architecture basis](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/traceability/research-basis.md)
- [MLS2PLM canonical equations](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/papers/mls2plm-canonical-equations.md)
- [Release acceptance guide](https://github.com/ContextualWisdomLab/fast-mlsirm/blob/main/docs/release_acceptance.md)

## Evidence boundary

Treat protected-main source, immutable releases, exact artifact provenance, and model-specific validation evidence as authoritative. An open pull request or documentation page is not evidence that a method, estimator, calibration policy, accelerator path, or package version has shipped.

Scientific claims remain tied to the repository's cited literature, accepted research-to-architecture decisions, canonical equations, and executable validation rather than generic quality language.
