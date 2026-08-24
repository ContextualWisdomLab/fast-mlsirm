# Reproducible PyPI release publishing

## Added

- Added release-tag-bound sdist and wheel publication with a project-version provenance check, pinned Maturin and PyPA publisher revisions, and persisted checkout credentials disabled.
- The canonical release-tag workflow now explicitly dispatches package publication from the immutable tag, avoiding reliance on release events created with `GITHUB_TOKEN`, which do not recursively start ordinary event-triggered workflows.
- Isolated GitHub release-asset mutation from PyPI credentials, removed the unpinned runtime Twine installation path, and kept duplicate GitHub release assets and PyPI filenames fail-closed rather than silently replacing an immutable release artifact.
- PyPI publication now depends directly on the verified build artifacts rather than successful GitHub asset attachment, so a failed PyPI publication can be retried even when immutable release assets already exist and correctly reject replacement.
