# Group Cargo updates across package and fuzz lock roots

## Changed

- Manage the root, standalone PyO3, and fuzz Cargo lock roots through one Dependabot directories lane. Version updates are grouped by dependency name and security updates use a dedicated security-update group so independently locked roots do not silently diverge during repository-owned dependency maintenance.
