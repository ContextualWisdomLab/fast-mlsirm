# Release license evidence and NumPy lock reconciliation

## Changed

- `uv.lock`, `requirements/ci.txt` and `requirements/package.txt` all pin
  numpy 2.5.3. Before this change they pinned 2.5.1 and 2.5.2, and the audit
  environment ran 2.5.3. Only the numpy entries changed. The requirements
  files carry the complete 2.5.3 PyPI hash set. The user-facing
  `numpy>=1.24` constraint is unchanged.

## Added

- `docs/security/license-evidence-0.11.5.md` records which artifacts are
  published: the PyPI sdist and 12 wheels. The Rust crates are not published.
  It also records the per-dependency license inventory for both ecosystems and
  three verdicts:
  - atheris 3.1.0 is Apache-2.0. PyPI metadata declares no license, so this
    is determined from its LICENSE file, which is hash-matched to upstream.
  - r-efi 5.3.0 and 6.0.0 are used under an explicit MIT election with a
    recorded rationale. They are reached only through a dev-dependency path.
  - The official NumPy wheels bundle libgfortran and libquadmath. libquadmath
    is LGPL-2.1-or-later and has no runtime exception, so this is marked as
    an owner decision, not approved.
- `tools/license_inventory.py` is an offline generator for
  `docs/security/license-evidence-0.11.5/inventory.json`.
