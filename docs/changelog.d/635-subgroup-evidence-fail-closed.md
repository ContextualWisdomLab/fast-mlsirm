# Subgroup validation evidence fails closed

## Fixed

- Automated-scoring subgroup SMD gates reject requested subgroups with fewer
  than two paired cases or zero human variance instead of silently skipping
  them and reporting a vacuous pass.
