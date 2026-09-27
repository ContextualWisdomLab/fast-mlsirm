# Central reviewed artifact texts

Reviewed central merge `5b0024a9` (#2457): complete MIT grants for arbitrary 1.4.2, libfuzzer-sys 0.4.13 and shlex 2.0.1. Four license member bodies, including shlex’s Apache application notice, exactly match the independently downloaded official crate archives and recorded SHA256s. The notice alone is not a complete Apache grant: acceptance requires the independent complete MIT alternative and explicit MIT selection.

The reviewed artifact text suite passed 762 tests against central integration source `3e01ce3f` (containing #2457). Fast release callers still pin central `422defa2`; this review does not change their runtime callee. It does not resolve the original objc2/Android/Python HOLD baseline or establish published-wheel acceptance.
