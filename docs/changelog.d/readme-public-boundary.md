# README public-description boundary

## Changed

- `README.md` is the PyPI `long_description`, so it no longer carries internal
  commercial-boundary vocabulary. The `Commercial Readiness` section — with the
  enterprise sales gate, the KRW 2,000,000,000 product gate, buyer packet,
  procurement due-diligence, PR queue governance, Figma evidence sync links, and
  the multi-step evidence build transcript — is replaced by a `Project Status`
  section that states scope, release verification, and the security, support,
  changelog, and ADR entry points. The same evidence machinery is unchanged and
  stays documented in `docs/commercial_readiness.md` and
  `docs/release_acceptance.md`.
- Repo-relative README links now resolve to absolute GitHub URLs. Only `LICENSE`
  and the Python sources ship in the distribution, so `docs/`, `SECURITY.md`,
  `SUPPORT.md`, and `CHANGELOG.md` links were dead on the PyPI project page.

## Fixed

- `scripts/sales_readiness.py` gained a `public_boundary:README.md` check that
  fails the gate when internal commercial, procurement, buyer, or monetary-target
  vocabulary reappears in the published package description. The required
  commercial tokens it used to demand from `README.md` are now required in
  `docs/commercial_readiness.md` and `docs/enterprise_sales_readiness.md`, where
  that language belongs.
