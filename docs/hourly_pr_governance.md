# Hourly PR Queue Governance

The repository runs the existing PR queue governance evidence builder at the top of every hour and on manual dispatch. The scheduled workflow is deliberately read-only: it records open pull requests, review/check delays, changes requested, staleness, duplicate-looking scope, and release-scope conflicts, then uploads the JSON manifest and accessible HTML report as retained GitHub Actions artifacts.

Native branch protection and explicitly enabled GitHub auto-merge remain responsible for merging reviewed pull requests after required checks pass. The scheduled workflow does not modify source, bypass reviews, weaken checks, or grant itself write permissions.
