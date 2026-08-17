# Bounded JSON loads for release acceptance

## Security

- Release acceptance and generation-request contract loading use size- and
  depth-bounded JSON parsers instead of unbounded `json.loads` on CLI stdout
  and fit_summary artifacts.
