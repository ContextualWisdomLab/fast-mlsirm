# Release changelog serialization authority

## Purpose

Document the authority split between append-only fragment authorship on feature branches and fail-closed aggregate verification at immutable release publication.

## Design decision

Feature pull requests contribute authoritative notes under `docs/changelog.d/*.md`. The derived Unreleased block in `CHANGELOG.md` is serialized and checked at release-tag time (`scripts/render_changelog_fragments.py --check`) immediately before tag-state classification and immutable tag creation. Ordinary feature CI retains fragment format, determinism, and renderer round-trip contracts without requiring every branch to rewrite the shared derived aggregate.

## References

Keep a Change Log. (n.d.). *Keep a Changelog*. https://keepachangelog.com/

Semantic Versioning 2.0.0. (n.d.). *Semantic Versioning*. https://semver.org/
