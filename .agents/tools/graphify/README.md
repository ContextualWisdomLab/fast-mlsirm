# Local Graphify engine

Graphify 0.9.58 resolves bare Python callbacks through a global name map,
misbinding same-name nested functions. `engine.patch` adds lexical lookup only
for identifier references; the existing attribute-name heuristic is preserved.
This is the project-local workaround for #1862, not an upstream release.

Run from the repository root:

```sh
uv venv --python 3.14 .agents/tools/graphify/.venv
uv pip install --python .agents/tools/graphify/.venv/bin/python --require-hashes -r .agents/tools/graphify/requirements.txt
.agents/tools/graphify/.venv/bin/python .agents/tools/graphify/apply_patch.py
.agents/tools/graphify/.venv/bin/python .agents/tools/graphify/test_apply_patch.py -v
.agents/tools/graphify/.venv/bin/python .agents/tools/graphify/test_callback_scope.py -v
.agents/tools/graphify/.venv/bin/graphify update . --no-cluster
```

The helper requires its sibling `.venv`, rejects symlinks escaping that runtime
and unknown source/patch hashes, stages native `patch` output, verifies the
result, and atomically replaces only the local engine. Reapplying is idempotent.
`manifest.json` identifies the official wheel, source, patch, and resulting
engine; wheel integrity is enforced by the hashed installation command.
The environment contains only the tool's required AST dependencies, with no
model-provider extras. This does not configure Graphify MCP or a hosted CI gate.

Local validation: the unpatched tool fails the sibling-callback case while the
cross-file and existing attribute-heuristic cases pass; the patched tool passes
all three. Nine helper safety cases pass. On the #1599 three-file integration
fixture, exactly one erroneous target changes and all other extracted edges are
preserved. These are graph extraction checks, not runtime parser proofs.

Restore the pinned upstream package with:

```sh
uv pip install --python .agents/tools/graphify/.venv/bin/python --force-reinstall --require-hashes -r .agents/tools/graphify/requirements.txt
```

When an upstream release fixes #1862, update the lock and provenance, run the
same regressions without the patch, then remove the patch/apply helper. Do not
reuse the old AST cache across extractor changes. License and NOTICE files from
the official wheel are retained in `licenses/`.
