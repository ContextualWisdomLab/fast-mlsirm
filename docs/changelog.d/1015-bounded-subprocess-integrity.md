# Harden bounded subprocess cleanup

## Fixed

- Keep governance and procurement subprocess capture bounded across stdout, stderr, execution time, decoding, and JSON parsing. POSIX cleanup now avoids re-signalling an already reaped process group, successful capture closes parent-side pipe descriptors deterministically, and timeout/overflow paths retain fail-closed evidence without weakening repository gates.
