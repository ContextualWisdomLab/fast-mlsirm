# CodeQL action release identity

## Changed

- Updated repository-owned CodeQL `init` and `analyze` phases together to the reviewed immutable `github/codeql-action` v4.37.9 commit `cdf488f595d80d6e07e03d4674febd5ab45fa938`.
- Strengthened the workflow contract so both Actions-language and manual Python jobs must keep `init` and `analyze` on that same exact commit, preventing split Dependabot updates from creating mixed CodeQL action releases.
