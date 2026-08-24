# Bind releases and package artifacts to reviewed source commits

## Fixed

- Require manual release publication to name the exact reviewed release source commit, prove that commit is on the current protected default-branch lineage and is the commit that introduced both the requested project version and its released CHANGELOG section relative to its first parent, validate release metadata from that commit, and create or resume the immutable version tag only when it targets that same commit. This prevents either a later default-branch commit or an unrelated same-version descendant from being silently included in an already-cut release.
- Carry that same canonical release commit into package publication, verify the immutable version tag still peels to it, and build every sdist and wheel from the explicit commit rather than independently resolving the tag. The package workflow itself is dispatched from the protected default branch so an older release tag cannot select a predecessor publication-control definition.
