# Item-bank lifecycle public identity replay

## Fixed

- Governed item-bank lifecycle records now replay their factory-sealed creation-time identity before returning a public fingerprint/id or serializing JSON-compatible content. A package-owned weak creation-seal registry binds each live factory-created object identity to its original fingerprint, so coherently rebinding both lifecycle content and the record's stored digest cannot manufacture fresh authority; dead-record entries are removed without retaining the record, and object-identity reuse is rejected. Callback-bearing mutations continue to fail closed, while valid lifecycle identities and payloads remain unchanged. No calibration, fit, DIF, information, linking, scoring, uncertainty, or other psychometric arithmetic changes.
