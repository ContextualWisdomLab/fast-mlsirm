"""Generate compile-time whole Rust/WGSL/dependency source binding; no fit.

The generated observer excludes itself (no recursive hash) and is bound by the
frozen source packet and actual native executable. Cargo.lock pins dependencies;
compiler/native and the complete reviewed tree are separate artifact bindings.
"""
from pathlib import Path
import json
W=Path(__file__).resolve().parents[1]
paths=sorted(p for p in W.rglob('*') if p.is_file()
 and not any(x in ('.git','.codegraph','target','target-official','__pycache__') or x.startswith('target-') for x in p.parts)
 and (p.suffix in ('.rs','.wgsl') or p.name in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'))
 and p.name!='mask_grm_resume_sources.rs')
lines=['//! Compile-time complete numerical Rust/WGSL and Cargo dependency binding.',
 'use sha2::{Digest, Sha256};',
 'pub fn current() -> serde_json::Value {',
 '    let mut out = serde_json::Map::new();']
base=W/'crates/mlsirm-core/src'
import os
for p in paths:
 rel=str(p.relative_to(W));inc=os.path.relpath(p,base)
 lines.append(f'    out.insert({json.dumps(rel)}.into(), serde_json::json!(format!("{{:x}}", Sha256::digest(include_bytes!({json.dumps(inc)})))));')
lines+=['    serde_json::Value::Object(out)','}']
out=base/'mask_grm_resume_sources.rs'
# This owned generated source is deterministic and never edits solver bytes.
out.write_text('\n'.join(lines)+'\n')
print(json.dumps({'generated':str(out),'bound_paths':len(paths)}))
