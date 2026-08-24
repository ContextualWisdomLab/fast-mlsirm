# Seal bounded JSON semantic-input callback boundaries

## Fixed

- Reject caller-defined byte/depth limit integers before comparison and caller-defined JSON text subclasses before encoding, while preserving exact built-in controls, bounded parsing semantics, and the existing descriptor/path/size/depth defenses used by repository release and governance automation.
