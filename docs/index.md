# fast-mlsirm

fast-mlsirm is a high-performance psychometric measurement toolkit centered on multidimensional latent-space item-response modeling, simulation, fitting, recovery diagnostics, calibration, and evidence-bound evaluation workflows.

## Start here

Use fast-mlsirm when measurement or evaluation work needs explicit model assumptions, reproducible estimation, calibration evidence, and a production numerical path that keeps performance-critical mathematics in the Rust core while exposing a practical Python package surface.

The repository includes MLSIRM/MLS2PLM work, item-response and calibration utilities, model/evaluation diagnostics, sampling and statistical evidence contracts, and integration surfaces used by other ContextualWisdomLab products.

## Documentation

- [Repository overview](../README.md)
- [Architecture](../ARCHITECTURE.md)
- [Product/technical gap baseline](product-technical-gap-baseline.md)
- [Architecture decisions](adr/)
- [Research and method notes](research/)
- [Release guidance](releasing.md)

## Evidence boundary

Treat protected-main source, immutable releases, exact artifact provenance, and model-specific validation evidence as authoritative. An open pull request or documentation page is not evidence that a method, estimator, calibration policy, accelerator path, or package version has shipped.

Scientific claims should remain tied to the repository's cited literature and executable validation rather than generic quality language.
