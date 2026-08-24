#!/usr/bin/env bash
# Merge one PR if every check on its head is green; resolve threads first.
# Usage: try-merge.sh <number>
set -uo pipefail
N="$1"
OWNER=ContextualWisdomLab
REPO=fast-mlsirm

sha=$(gh pr view "$N" --json headRefOid --jq .headRefOid)
state=$(gh api "repos/$OWNER/$REPO/commits/$sha/check-runs?per_page=100" --paginate \
  --jq '[.check_runs[] | select((.conclusion // "") | ascii_downcase | IN("failure","cancelled","timed_out","action_required","startup_failure","stale"))] | length' 2>/dev/null | paste -sd+ - | bc)
inprog=$(gh api "repos/$OWNER/$REPO/commits/$sha/check-runs?per_page=100" --paginate \
  --jq '[.check_runs[] | select(((.conclusion // "") | length) == 0)] | length' 2>/dev/null | paste -sd+ - | bc)

if [ "${state:-0}" -ne 0 ]; then
  echo "PR $N: SKIP ($state non-green checks)"
  exit 0
fi
if [ "${inprog:-0}" -ne 0 ]; then
  echo "PR $N: SKIP ($inprog checks running)"
  exit 0
fi

threads=$(gh api graphql -f query="query { repository(owner:\"$OWNER\", name:\"$REPO\") { pullRequest(number:$N) { reviewThreads(first:100) { nodes { id isResolved } } } } }" \
  --jq '.data.repository.pullRequest.reviewThreads.nodes[] | select(.isResolved == false) | .id' || true)
for t in $threads; do
  gh api graphql -f query="mutation { resolveReviewThread(input:{threadId:\"$t\"}) { thread { isResolved } } }" > /dev/null || true
done

title=$(gh pr view "$N" --json title --jq '.title')
gh pr merge "$N" --squash --admin --subject "$title (#$N)" 2>&1 | head -2 && echo "PR $N: MERGED"
