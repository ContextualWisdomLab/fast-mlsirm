# Git metadata lookup deadlines

## Fixed

- Commercial evidence builders fail closed with a bounded Git metadata timeout
  (benchmark, buyer packet, procurement, commercial release, Figma evidence)
  so hung `git rev-parse` cannot hang release pipelines.
