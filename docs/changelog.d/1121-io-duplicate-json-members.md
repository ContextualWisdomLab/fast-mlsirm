# Reject ambiguous duplicate JSON artifact members

## Fixed

- The shared bounded artifact JSON loader now rejects duplicate object member
  names at every nesting level instead of accepting last-value-wins semantics,
  while preserving its existing stable-file, UTF-8, byte, nesting, and parser
  controls.
