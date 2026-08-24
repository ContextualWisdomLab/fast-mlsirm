# Restore semantic essay table row headers

## Fixed

- Mark the identity axis of governed essay facets-calibration and validation-evidence tables with explicit `<th scope="row">` semantics. Task, rater, respondent, category/iteration, and validation-metric identities now remain programmatically associated with their row while numerical scoring and calibration arithmetic remain unchanged.
- Preserve complete table and canonical-JSON evidence when standalone reports are printed or exported to PDF by removing screen-only scroll clipping and the JSON height cap in print media.
