# Hourly PR queue governance

## Added

- A read-only hourly GitHub Actions loop that runs the existing PR queue governance evidence builder, publishes its JSON and HTML audit artifacts, and retains native branch-protection and auto-merge gates instead of bypassing review or checks.
