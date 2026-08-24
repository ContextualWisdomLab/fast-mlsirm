# Bind release tags to reviewed source commits

## Fixed

- Require manual release publication to name the exact reviewed release source commit, verify that commit is on the current protected default-branch lineage, validate release metadata from that commit, and create or resume the immutable version tag only when it targets that same commit. This prevents a later default-branch commit from being silently included in an already-cut release.
