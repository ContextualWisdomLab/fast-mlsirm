# WebGL historical XML rights clarification draft

Status: local draft, not sent.

Proposed destination: KhronosGroup/WebGL issue or the licensing contact identified by its maintainers.

We are auditing the XML embedded in khronos_api 3.1.0. Its publisher-linked WebGL commit is c987f075bfdca44119175c41f547d849c08983a3. We verified all bundled files against that immutable commit. The root LICENSE.txt was added in ca07c9628d207b5cc560ee6eac32ceeb1c39a2ed on March 30, 2019. Forty-seven of the 55 bundled extension XML files are byte-identical at that licensed commit. Eight differ, listed below.

Could you confirm whether the repository's MIT/Materials permission covers the earlier versions of these eight files at c987f075bfdca44119175c41f547d849c08983a3? If yes, please identify the authoritative grant and required copyright/permission notices for redistribution of these exact earlier versions, including their embedded IDL and specification text. We would preserve those notices alongside the package's Apache and ANGLE BSD notices.

Files:

- extensions/EXT_float_blend/extension.xml
- extensions/EXT_texture_compression_bptc/extension.xml
- extensions/EXT_texture_compression_rgtc/extension.xml
- extensions/KHR_parallel_shader_compile/extension.xml
- extensions/WEBGL_draw_buffers/extension.xml
- extensions/WEBGL_multiview/extension.xml
- extensions/proposals/EXT_multi_draw_arrays/extension.xml
- extensions/proposals/WEBGL_blend_equation_advanced_coherent/extension.xml

No requested relicensing, feature removal, or waiver. The per-file SHA256 and original source URLs are in webgl-first-grant-file-correspondence.json and khronos-submodule-source-correspondence.json.
