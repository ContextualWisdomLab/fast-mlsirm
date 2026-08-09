# Doctoring record: bounded retries for hourly PR governance

## Decision

The hourly PR-governance workflow retries only failed GitHub snapshot evidence caused exclusively by explicit HTTP 502, 503, or 504 responses. Retries are bounded to three total attempts and use deterministic exponential delays of 10 and 20 seconds. Every attempt removes the prior output directory so stale evidence cannot authorize a later success.

HTTP 429 is deliberately not retried by this short-delay mechanism. The governance manifest currently preserves command stderr and return codes but does not retain the response headers needed to honor GitHub's `Retry-After` or `X-RateLimit-Reset` instructions. A subsequent hourly run provides a naturally bounded retry without risking repeated requests during an active rate limit.

HTTP 500, generic timeout text, connection resets, DNS failures, and broad “temporarily unavailable” phrases remain non-retryable. Those markers are insufficiently specific to distinguish a safe transient GitHub server failure from client, network, authentication, or implementation defects. Governance failures, contradictory manifests, malformed evidence, successful return codes recorded as errors, and Boolean pseudo-return-codes also fail closed.

## Standards and platform rationale

GitHub documents primary and secondary rate-limit responses as HTTP 403 or 429 and requires clients to honor `Retry-After` when present, wait until `X-RateLimit-Reset` when the remaining quota is zero, or otherwise wait at least one minute before retrying. Because the current manifest does not preserve those headers, a fixed 10-second retry for HTTP 429 would not satisfy that contract.

GitHub also recommends exponential waiting for repeated secondary-rate-limit failures and terminating after a bounded number of retries. The workflow applies exponential waiting only to the separately allowlisted 502/503/504 server responses, where no rate-limit header contract is implicated.

## Verification contract

`tests/test_hourly_pr_governance_workflow.py` extracts and executes the embedded classifier from the actual YAML workflow. The suite proves that:

- only explicit HTTP 502, 503, and 504 errors are retryable;
- HTTP 429 is rejected without rate-limit response-header evidence;
- real governance failures cannot be hidden by a simultaneous server error;
- malformed, contradictory, or success-coded error evidence is rejected;
- retries remain bounded and exponentially delayed;
- the workflow stays read-only, time-bounded, and source-reviewable; and
- the de-indented shell block is syntactically valid Bash.

These tests verify the repository contract and do not claim availability guarantees for GitHub's external service.

## Compatibility and rollback

The change affects only retry policy for a read-only evidence workflow. It does not alter pull requests, repository contents, branch protection, review policy, psychometric arithmetic, database objects, package APIs, or release artifacts. Rollback would restore broader retry matching, but would reintroduce the risk of retrying rate-limited or ambiguous failures without the required timing evidence.

## References

GitHub. (2026). *Best practices for using the REST API*. GitHub Docs. Retrieved August 7, 2026, from https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api

GitHub. (2026). *Rate limits for the REST API*. GitHub Docs. Retrieved August 7, 2026, from https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api

GitHub. (2026). *Troubleshooting the REST API*. GitHub Docs. Retrieved August 7, 2026, from https://docs.github.com/en/rest/using-the-rest-api/troubleshooting-the-rest-api
