# GitHub GraphQL query bounds for hourly PR governance

## Operational finding

The hourly governance workflow previously requested every open pull request plus nested changed files, labels, merge state, and review state in one `gh pr list --limit 100` query. With 37 open pull requests, GitHub ended that request with `GraphQL: Resource limits for this query exceeded`. The derived manifest then failed closed but showed `open_pr_count: 0`, which made the evidence incomplete and prevented the workflow-registry audit.

## Adopted design

The workflow now uses a split-query snapshot:

1. Enumerate at most 101 open pull-request identities using only `number`.
2. Reject the snapshot if more than the supported 100 open pull requests are observed.
3. Enrich each admitted identity with one bounded `gh pr view` request containing the existing classification fields.
4. Exclude a pull request if its detailed state is no longer `OPEN`.
5. Require every open detail payload to retain the requested body, head/base identity, review, merge-state, label, changed-file, lifecycle, and URL fields before promotion into complete evidence.
6. Bound the complete live capture with a 420-second monotonic deadline so sequential enrichment cannot consume the workflow's ten-minute job budget; budget exhaustion is emitted as explicit fail-closed evidence rather than allowing job-level cancellation to erase the snapshot.
7. Preserve the existing light, bounded history query and exact default-branch SHA lookup when the capture budget remains available.
8. Publish the raw snapshot beside the deterministic governance manifest and HTML report.

This follows GitHub's guidance to reduce nested query depth, request only required fields, use smaller collections, paginate, and split large queries. It also follows the GitHub CLI contract that GraphQL pagination requires an `endCursor` variable and `pageInfo`; split enrichment was selected here because it preserves the existing `gh pr view` field semantics while keeping each request independently bounded.

## Security and reliability boundary

- The workflow remains read-only.
- GitHub commands have per-command deadlines and the complete live capture has a cumulative wall-clock deadline.
- Only explicit HTTP 502, 503, and 504 responses are retried.
- Authentication, rate-limit, malformed JSON, timeout, cumulative-budget, duplicate identity, detail mismatch, incomplete detail, and queue-cap failures remain fail-closed.
- No successful or partial response is promoted into complete evidence when any required open-PR detail is missing.

## References

GitHub. (n.d.). *GitHub CLI manual: gh api*. Retrieved August 15, 2026, from https://cli.github.com/manual/gh_api

GitHub. (n.d.). *Rate limits and query limits for the GraphQL API*. Retrieved August 15, 2026, from https://docs.github.com/en/graphql/overview/rate-limits-and-query-limits-for-the-graphql-api

GitHub. (n.d.). *Using pagination in the GraphQL API*. Retrieved August 15, 2026, from https://docs.github.com/en/graphql/guides/using-pagination-in-the-graphql-api
