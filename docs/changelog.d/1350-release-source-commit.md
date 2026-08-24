# Bind release tags to reviewed source commits

## Fixed

- Require manual release publication to name the exact reviewed release source commit, prove that commit is on the current protected default-branch lineage and is the commit that introduced both the requested project version and its released CHANGELOG section relative to its first parent, validate release metadata from that commit, and create or resume the immutable version tag only when it targets that same commit. This prevents either a later default-branch commit or an unrelated same-version descendant from being silently included in an already-cut release.
