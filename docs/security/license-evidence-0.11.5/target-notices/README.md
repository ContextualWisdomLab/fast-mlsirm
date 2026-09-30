# Target-specific third-party notices for 0.11.5 candidates

These files were rendered from the hash-bound target inventories with
`tools/third_party_licenses.py` (SHA256
`eb173ffae71c6a9e0dc45a80bda19df5e68bde71dff9c17ca9acf2fb86f606fa`).
The source fixture SHA256 is
`57930c0f32ff995895a0599fc1fd162f1c7139055013bfab7238dd250c862f78`;
the upstream evidence manifest SHA256 is
`11229c9a7f0b5eaaa2ddec3990ef6d35201fa6f53ec88694d8dab388e2424529`.
The binding `Cargo.lock` SHA256 is
`528414b582b256a6ab8ac68ceceb134e8ddb719774513ab8cc3a0e4eb22a177d`.

| Target | Entries | Snapshot SHA256 | Notice SHA256 |
| --- | ---: | --- | --- |
| Linux aarch64 | 101 | `23452c4e7daaf7b36cbf2036ab4d1395d106b7bc5cfcef2bb4572d24ac4b476b` | `afa61d22b98ec3b8d4797c4af1dc9a6d159b2051a543e658c67f4d2404d2d87c` |
| Windows x86_64 | 117 | `d39171f014f4b3603a564d2d9339e72d4fd5166b98714f7945dc8c727c69ed96` | `7a3537a1df4063760bfee6e720b7cc9105721748d3fffd29aed2732d155dde58` |

The existing root `LICENSE-THIRD-PARTY` covers Linux x86_64 only. macOS
universal2 has no reviewed bundle: its target inventory still has ten `objc2`
family HOLD rows. The wheel workflow stops before building that target.
Reproduction inputs and outputs are retained on s1 under
`/data/orca/workspaces/fmls-license-evidence/target-notices-20260926/`.
