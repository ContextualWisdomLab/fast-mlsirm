# Seal bounded subprocess command admission

## Fixed

- Reject caller-defined command-container and text-token subclasses before repository automation materializes or checks command arguments, preventing validation-time callback execution while preserving exact built-in list and tuple vectors.
