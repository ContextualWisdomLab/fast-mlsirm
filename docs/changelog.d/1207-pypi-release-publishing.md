# Reproducible PyPI release publishing

## Added

- Added release-tag-bound sdist and wheel publication with a project-version provenance check, pinned Maturin and PyPA publisher revisions, and persisted checkout credentials disabled.
- Isolated GitHub release-asset mutation from PyPI credentials, removed the unpinned runtime Twine installation path, and kept duplicate GitHub release assets and PyPI filenames fail-closed rather than silently replacing an immutable release artifact.
