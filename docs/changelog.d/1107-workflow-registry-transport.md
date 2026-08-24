# Workflow registry transport failures

## Fixed

- The read-only workflow-registry audit now converts missing or inaccessible local GitHub CLI execution into a stable fail-closed `GitHubApiError`, so automation can emit bounded failure evidence instead of crashing with raw operating-system details.
