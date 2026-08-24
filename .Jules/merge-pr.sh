#!/usr/bin/env bash
# Resolve all unresolved review threads on a PR, then squash-merge it.
# Usage: merge-pr.sh <number> [commit-title]
set -euo pipefail
N="$1"
TITLE="${2:-}"
OWNER=ContextualWisdomLab
REPO=fast-mlsirm

threads=$(gh api graphql -f query="query { repository(owner:\"$OWNER\", name:\"$REPO\") { pullRequest(number:$N) { reviewThreads(first:100) { nodes { id isResolved } } } } }" --jq ".data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .id")
for t in $threads; do
  gh api graphql -f query="mutation { resolveReviewThread(input:{threadId:\"$t\"}) { thread { isResolved } } }" > /dev/null && echo "resolved $t"
done

if [ -z "$TITLE" ]; then
  TITLE="$(gh pr view "$N" --json title --jq '.title') (#$N)"
fi
gh pr merge "$N" --squash --admin --subject "$TITLE"
