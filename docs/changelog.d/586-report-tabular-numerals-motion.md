# Diagnostics-report numeric alignment and motion cleanup

## Changed

- Applied tabular numeral styling to standalone diagnostics-report body text so numeric values can align more consistently when the selected font supports equal-width figures.
- Removed obsolete opacity transitions from bar rows and table rows while preserving the active table-row background hover cue and the existing reduced-motion override.
- Added a rendered-report regression that pins the numeric-style declaration, transition cleanup, hover cue, and reduced-motion contract without changing report data, score semantics, or exported exact values.
