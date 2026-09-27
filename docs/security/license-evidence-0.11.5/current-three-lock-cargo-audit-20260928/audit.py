"""Reuse the deployed archive/notice/license evaluator on all three current locks."""
import collections, hashlib, io, json, pathlib, subprocess, sys, tarfile, tomllib
source = pathlib.Path('/tmp/fmls-current-license-full-20260928')
helper = pathlib.Path('/tmp/fmls-release-sidecar-runtime-adoption-20260928')
sha = 'f54143b968c0f332fa8d49ed3c116fc7d51891b0'
helper_sha = 'e45f1b144aef900d734ff4c900f9e0010fd5a32d'
assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == sha
subprocess.run(['git','-C',str(helper),'diff','--exit-code',helper_sha,'--','scripts/ci'],check=True)
sys.path.insert(0,str(helper/'scripts/ci'))
import release_dependency_gate as gate

def blob(path):
    data = subprocess.check_output(['git','-C',str(source),'show',f'{sha}:{path}'])
    assert (source/path).read_bytes() == data
    return data

choices = json.loads(blob('docs/release-license-selections.json'))
selections = {(r['name'],r['version']):r for r in choices if r['ecosystem']=='cargo'}
assert len(selections) == sum(r['ecosystem']=='cargo' for r in choices)
packages, locks = {}, {}
for path in ('Cargo.lock','crates/fast-mlsirm-py/Cargo.lock','fuzz/Cargo.lock'):
    data = blob(path); locks[path] = hashlib.sha256(data).hexdigest()
    for pkg in tomllib.loads(data.decode())['package']:
        if not pkg.get('source'): continue
        assert pkg['source'] == 'registry+https://github.com/rust-lang/crates.io-index'
        key = pkg['name'],pkg['version']
        if key in packages: assert packages[key]['checksum'] == pkg['checksum']
        packages[key] = pkg
caches = list((pathlib.Path.home()/'.cargo/registry/cache').glob('*'))
rows = []
for key,pkg in sorted(packages.items()):
    name,version = key; subject = f'cargo/{name}@{version}'
    archive = next((c/f'{name}-{version}.crate' for c in caches if (c/f'{name}-{version}.crate').is_file()),None)
    assert archive is not None, subject
    raw = gate.read_archive_snapshot(archive)
    digest = hashlib.sha256(raw).hexdigest(); assert digest == pkg['checksum'], subject
    bound = gate.archive_license_evidence(raw,'cargo')
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        manifest = tomllib.loads(tar.extractfile(f'{name}-{version}/Cargo.toml').read().decode())['package']
    assert (manifest['name'],manifest['version']) == key
    selection = selections.get(key)
    if selection: assert selection['archive_sha256'] == digest, subject
    evidence = {**bound,'ecosystem':'cargo','license':manifest.get('license')}
    texts,notice = gate._source_license_notice(source,sha,subject,evidence,selection)
    evidence['license_texts'] = {**bound['license_texts'],**texts}
    failures,decision,license_source = gate.evaluate_dependency_license(evidence,subject,selection)
    rows.append(dict(subject=subject,archive_sha256=digest,source_url=f'https://static.crates.io/crates/{name}/{name}-{version}.crate',declared_license=manifest.get('license'),license_member_sha256=bound['license_member_sha256'],chosen=selection.get('chosen') if selection else None,source_notice=notice,allowed=decision.allowed and not failures,failures=[vars(f) for f in failures],license_source=license_source))
result = dict(schema='fmls-current-cargo-archive-audit/v1',source_sha=sha,helper_sha=helper_sha,lock_sha256=locks,selection_manifest_sha256=hashlib.sha256(blob('docs/release-license-selections.json')).hexdigest(),evaluator_sha256=hashlib.sha256((helper/'scripts/ci/release_dependency_gate.py').read_bytes()).hexdigest(),scope='All registry archives in the union of three current committed Cargo locks; archive/source-notice license evaluation only. Not graph closure, native-link/Strix, original historical HOLD clearance, or published wheel acceptance.',packages=len(rows),passed=sum(r['allowed'] for r in rows),failure_codes=dict(collections.Counter(f['code'] for r in rows for f in r['failures'])),rows=rows)
out=pathlib.Path('/tmp/fmls-current-cargo-archive-audit-20260928.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
for row in rows:
    if not row['allowed']: print(row['subject'],row['failures'])
