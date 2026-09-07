# Acquisition readiness documentation alignment

## Fixed

- Align the release, commercial, enterprise, governance, and README guidance
  with the current `--require-acquisition-readiness` profile.
- Mark the retained KRW 2,000,000,000 and `--require-20b-product` material as
  deprecated compatibility evidence, without treating readiness evidence as a
  valuation or transaction-price claim.
- Remove protected-merge bypass language from the enterprise readiness guide.
- Keep buyer-packet benchmark and release HTML evidence confined to the
  containing manifest evidence root after path resolution, so absolute paths,
  parent traversal, and symlink-resolved escapes cannot import unrelated files
  into procurement evidence.
- Replay each linked benchmark and release HTML SHA-256 against the digest
  declared by its containing manifest before admitting the file to the buyer
  packet, so in-root post-manifest tampering fails closed.
- Require supplied benchmark and release-evidence manifests to carry
  `source_commit` whenever the buyer packet is bound to an exact source
  revision, so optional evidence cannot be relabeled from an unidentified
  build while required acceptance and sales evidence remain source-bound.
