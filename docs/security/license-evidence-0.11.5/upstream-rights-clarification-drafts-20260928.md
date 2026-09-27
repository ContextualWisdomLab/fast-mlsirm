# Upstream source-rights clarification drafts

Status: local drafts only. No issue, comment, email or chat message has been sent. These requests concern immutable source and permission evidence; they do not request legal advice or a licence waiver.

## 1. madsmtm/objc2

Suggested title: Source licence and SDK provenance for published 0.6/0.3 framework bindings

We are checking the source and notice chain for these published crates before redistributing a downstream package:

- block2 0.6.2
- dispatch2 0.3.1
- objc2 0.6.4
- objc2-encode 4.1.0
- objc2-core-foundation, objc2-core-graphics, objc2-foundation, objc2-io-surface, objc2-metal and objc2-quartz-core, all 0.3.2

At the original crate commits, LICENSE.md links to the applicable licences and discusses uncertainty around Apple SDK-derived bindings. The separate full licence files on current main are absent at the four original commits we checked. We have not assumed that later files retroactively supply all original notices.

For the six framework crates, parent commit 7b1abfd750a2cacaea71d6a56ecfb83cb7de560b pins objc2-generated at 4c2d6fb86d17ed4b4b7e68e80994788b60521987. All 528 compared source files match the parent or that generated snapshot; 400 generated blobs also match the immutable Git trees. The parent documentation names Xcode 26.0.1. Packaged Cargo.lock and normalized Cargo metadata were not independently regenerated.

Existing discussions #826 and #836 are relevant: #826 identifies the later root-licence addition at ee9a7ada2131f5944b8750428e265c15632f2a19; #836 discusses archive omission and was closed without adding those files. We are not asking you to repeat that packaging work. The remaining question is applicability to the exact historical source and independent SDK inputs.

Could you point us to:

1. The complete publisher licence texts and original copyright notices applicable to these exact crate versions, including the applicability of later licence files.
2. An immutable record of the SDK/header inputs used for this generated commit, including applicable notices and source-specific permissions for redistribution of the generated Rust output.
3. Any existing publisher or rights-holder clarification addressing the SDK-derived source scope discussed in LICENSE.md.

We are preserving the published archives and have not removed notices or substituted a newer SDK. Source correspondence alone is not being treated as a permission grant.

Reference: https://github.com/madsmtm/objc2/blob/7b1abfd750a2cacaea71d6a56ecfb83cb7de560b/LICENSE.md

## 2. rust-mobile/ndk

Suggested title: Immutable header and notice provenance for ndk-sys 0.6.0+11769913

We are checking the source and notice chain of ndk-sys 0.6.0+11769913, archive SHA-256 ee6cda3051665f1fb8d9e08fc35c96d5a244fb1be711a03b71118828afc9a873, originating at commit 49bbbba16c58ff63cb8a0ad0eca5a9fb7ecaec25.

All 11 compared source/documentation/original-manifest files match that commit. We recovered the full MIT and Apache root licence texts, but have not treated them as sufficient proof for every generated Android header input.

The generator specifies build 11769913 and ndk_platform.tar.bz2. An unauthenticated metadata request to its declared internal producer endpoint returned HTTP 404. This observation does not establish that the artifact was deleted or is inaccessible through every route.

Is there a public immutable producer artifact, header source manifest, input checksum/notice inventory, or existing publisher statement identifying the exact inputs and applicable redistribution terms for the four generated architecture files? A newer stable NDK would not establish the provenance of these particular published bytes.

Reference: https://github.com/rust-mobile/ndk/tree/49bbbba16c58ff63cb8a0ad0eca5a9fb7ecaec25/ndk-sys

## 3. KhronosGroup/WebGL

Suggested title: Historical source-markup licence scope for eight 2018 extension XML files

We are checking the original WebGL markup included in khronos_api 3.1.0. It corresponds to WebGL commit c987f075bfdca44119175c41f547d849c08983a3 from 2018-11-01.

The root licence was introduced later at ca07c9628d207b5cc560ee6eac32ceeb1c39a2ed. We compared the original 57 XML/IDL inputs with that snapshot: 48 files match, seven changed and two are absent. We have not assumed that a later root licence resolves every historical input. The two IDL files contain their own complete publisher grants; this request concerns source XML markup, distinct from published HTML/PDF specification terms.

Could you identify the historical source-markup grant or an authoritative licence-scope statement applicable to these eight original paths? Please also clarify whether the later root grant covers the earlier byte-identical XML inputs; byte equality by itself does not establish that scope.

- extensions/EXT_float_blend/extension.xml
- extensions/EXT_texture_compression_bptc/extension.xml
- extensions/EXT_texture_compression_rgtc/extension.xml
- extensions/KHR_parallel_shader_compile/extension.xml
- extensions/WEBGL_draw_buffers/extension.xml
- extensions/WEBGL_multiview/extension.xml
- extensions/proposals/EXT_multi_draw_arrays/extension.xml
- extensions/proposals/WEBGL_blend_equation_advanced_coherent/extension.xml

We have not classified the old XML as necessarily restricted merely from the separate specification publication terms, nor used current licence text to assign historical source permissions without scope evidence.

Reference: https://github.com/KhronosGroup/WebGL/tree/c987f075bfdca44119175c41f547d849c08983a3/extensions

## Separate release constraints

These clarifications do not waive the current bundled GNU-family text policy for r-efi 5.3.0/6.0.0. They also do not replace exact-head hosted checks, real Strix evidence, or the complete authenticated 13-distribution release set. All release requirements remain open until independently proven.
