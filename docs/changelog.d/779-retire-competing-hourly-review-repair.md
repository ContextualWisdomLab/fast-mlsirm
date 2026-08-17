# Retire competing hourly review-repair caller

## Fixed

- Remove the repository-local hourly review-repair GitHub Actions caller so only
  the organization single-writer control plane schedules mutation loops, matching
  ADR-0013 continuous-execution governance after failed startup evidence for the
  local caller.
