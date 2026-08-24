#!/usr/bin/env bash
# Resolve all unresolved review threads on a PR, then enable auto-merge (squash).
# Usage: automerge-pr.sh <number>
set -euo pipefail
N="$1"
OWNER=ContextualWisdomLab
REPO=fast-mlsirm

threads=$(gh api graphql -f query="query { repository(owner:\"$OWNER\", name:\"$REPO\") { pullRequest(number:$N) { reviewThreads(first:100) { nodes { id isResolved } } } } }" --jq ".data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .id" || true)
for t in $threads; do
  gh api graphql -f query="mutation { resolveReviewThread(input:{threadId:\"$t\"}) { thread { isResolved } } }" > /dev/null && echo "resolved $t"
done

gh pr merge "$N" --squash --auto 2>&1 || true
