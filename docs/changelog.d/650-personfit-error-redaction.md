# Person-fit invalid-response error redaction

## Security

- Stopped reflecting caller-controlled invalid response values in `person_fit_np()` validation errors while preserving the failing matrix coordinate and the complete-data 0/1 response contract.
