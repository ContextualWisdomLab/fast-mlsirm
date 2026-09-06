# GitHub release-asset digest replay

## Problem

The draft-first release transaction can successfully upload package distributions and the standalone SPDX SBOM, then wait for the independent PyPI sink before final publication. During that interval the GitHub release is intentionally still mutable. Rechecking only `draft == true` immediately before publication therefore proves the release state, but does not prove that the attached asset set still contains the reviewed bytes.

## Constraint and alternatives

Publishing the GitHub release before PyPI verification is rejected because immutable-release enforcement would seal an incomplete transaction. Treating successful `gh release upload` as durable evidence is also rejected because draft assets remain mutable until publication. Downloading every remote asset again would prove bytes but adds unnecessary transfer and a second hashing path when GitHub already exposes an authenticated SHA-256 digest for each release asset.

The tag-name lookup endpoint is also rejected for this check. GitHub documents `GET /releases/tags/{tag}` as returning a published release, while authenticated release listing exposes draft releases to callers with push access. Final draft verification therefore discovers the exact draft through authenticated paginated release enumeration, obtains its numeric release ID, and enumerates that release's assets through the release-ID asset endpoint.

## Decision

`finalize-release` downloads the exact `dist-*` workflow artifacts and `release-sbom` artifact from the same workflow run, recomputes each local SHA-256 and byte size, then reads the authenticated draft release immediately before publication. Publication fails closed unless the remote release contains exactly the same filenames, every asset is in the `uploaded` state, every `digest` equals `sha256:<local SHA-256>`, and every byte size matches. Missing, duplicate, unexpected, digest-mismatched, size-mismatched, or non-uploaded assets prevent the draft-to-published transition.

This is an integrity replay at the final irreversible seam, not a claim that the GitHub REST read and subsequent publish mutation are an atomic compare-and-swap operation. GitHub does not expose such an asset-set precondition on release publication. The workflow narrows that residual concurrency window by placing the authenticated digest replay directly before `gh release edit --draft=false`; protected operational policy must still prevent unreviewed manual mutation of an in-flight release draft.

## Primary-source basis

GitHub. (2026). *REST API endpoints for releases*. GitHub Docs. https://docs.github.com/en/rest/releases/releases

Scope used here: authenticated release listing returns draft releases to users with push access, while `GET /releases/tags/{tag}` is documented as fetching a published release. The finalizer therefore does not use the tag-name lookup as draft evidence; it discovers exactly one same-tag, same-name draft from the authenticated paginated list and uses its numeric release ID for the asset query.

GitHub. (2026). *REST API endpoints for release assets*. GitHub Docs. https://docs.github.com/en/rest/releases/assets

Scope used here: the release-asset representation includes the asset `name`, upload `state`, byte `size`, and a content `digest` such as `sha256:2151...`. The finalizer uses those server-reported fields as remote integrity evidence and compares them with the same workflow artifacts that fed the draft asset-upload job.

GitHub. (2026). *Immutable releases*. GitHub Docs. https://docs.github.com/en/code-security/concepts/supply-chain-security/immutable-releases

Scope used here: assets are intentionally assembled while the release is a draft and become protected after publication. That makes final pre-publication asset replay necessary: successful draft upload is not itself durable immutable evidence.

Sources checked on 2026-09-06.

## Acceptance evidence

The source-level RED commit `c74741695cac3ccd1b59eea443d8669244760bcb` requires a finalizer contract that reads GitHub asset digests and rejects set drift before publication. Initial repair `bea0630c6226adf69aeb1b1c7844fb0db486b17a` added exact filename/digest/size replay but used the published-release tag lookup for a draft. Primary-source review caught that endpoint mismatch. Follow-up RED `e1f07cf53ee4672f0139977177c9b8b9ab2b7916` requires draft-visible authenticated release enumeration and release-ID asset enumeration; causal repair `a494478738680331e89fad614929d522dde58556` implements those endpoints while preserving the same fail-closed artifact comparison. Hosted checks on the exact governed head remain the landing authority.
