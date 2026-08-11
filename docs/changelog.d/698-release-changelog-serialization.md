# Release serializes derived changelog aggregation

## Changed

- Move fragment→`CHANGELOG.md` aggregate parity enforcement from ordinary feature CI into the immutable release-tag workflow so concurrent PRs no longer thrash a shared derived file while release publication still fail-closes on drift.
