"""Source-only contracts for the isolated CI routing requirement.

These contracts do not attest runner registration, physical isolation or actual
native/GPU execution. Those require the existing runner operator's receipts.
"""
from pathlib import Path
import re

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_JOBS = ("python-matrix", "python", "rust", "gpu-smoke", "fuzz", "package")
_CHECKOUT_JOBS = tuple(job for job in _JOBS if job != "python")


_JOB_ID = re.compile(r"  ([A-Za-z_][A-Za-z0-9_-]*):")
_JOB_KEY = re.compile(r"    ([a-z][a-z0-9_-]*):(?: .*)?")


def _assert_jobs_grammar(workflow: str) -> None:
    """Admit only plain job ids and unique plain job keys; fail closed otherwise.

    Below ``jobs:`` every line indented fewer than six spaces must be blank, a
    comment, an exact ``  <id>:`` job line or a ``    <key>:`` job key that is
    unique within its job. Quoted, explicit, tagged, flow or trailing-text job
    lines and duplicate job keys are rejected, so the literal job slicer cannot
    miss a job or a second ``runs-on``. Non-LF line breaks (CR, NEL, LS, PS) and
    other C0 controls are rejected anywhere in the workflow, because YAML 1.1
    readers treat NEL/LS/PS as line breaks that ``str.split("\\n")`` keeps inside
    one comment line. Bounded source contract, not a YAML parser.
    """
    assert not re.search(r"[\x00-\x08\x0b-\x1f\x7f\x85\u2028\u2029]", workflow), (
        "unsupported line-break or control character"
    )
    head, sep, body = workflow.partition("\njobs:\n")
    assert sep and "\njobs:" not in body, "unsupported jobs section"
    keys = None
    for line in body.split("\n"):
        indent = len(line) - len(line.lstrip(" "))
        if line.strip(" ") == "" or indent >= 6 or line.lstrip(" ").startswith("#"):
            assert "\t" not in line[:indent + 1], "tab indentation is unsupported"
            continue
        if job := _JOB_ID.fullmatch(line):
            keys = set()
            continue
        key = _JOB_KEY.fullmatch(line)
        assert key and keys is not None, "unsupported job syntax"
        assert key.group(1) not in keys, "duplicate job key"
        keys.add(key.group(1))


def _job(workflow: str, name: str) -> str:
    """Read one literal top-level job from the retained CI workflow grammar."""
    _assert_jobs_grammar(workflow)
    pattern = rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
    matches = re.findall(pattern, workflow, re.MULTILINE | re.DOTALL)
    assert len(matches) == 1, f"expected exactly one CI job: {name}"
    return matches[0]


@pytest.mark.parametrize("name", _JOBS)
def test_ci_jobs_require_isolated_linux_x64_self_hosted_routing(name: str) -> None:
    """Keep every current CI job off hosted/control/scanner fallback routes."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    selectors = re.findall(r"^    runs-on: (.+)$", job, re.MULTILINE)
    assert selectors == ["[self-hosted, Linux, X64, cwlab-ci-isolated]"], (
        f"{name} must request the operator-provisioned isolated pool, without fallback"
    )


_CANONICAL_USES = re.compile(r"[a-z0-9_.-]+/[a-z0-9_.-]+(?:/[a-z0-9_./-]+)?@[0-9a-f]{40}")
_STEP = re.compile(r"      - ([a-z][a-z0-9_-]*):(.*)")
_STEP_KEY = re.compile(r"        ([a-z][a-z0-9_-]*):(.*)")
_ENTRY = re.compile(r"          ([A-Za-z_][A-Za-z0-9_-]*):(.*)")
_QUOTED = re.compile(r"(?:\"(?:[^\"\\]|\\.)*\"|'(?:[^']|'')*')(?:[ \t]+#.*)?")
def _value(rest: str) -> str:
    """Return one literal scalar after a key; reject unsupported separators and quoted continuations."""
    assert rest == "" or rest[0] in " \t", "unsupported key separator"
    value = rest.strip(" \t")
    if value[:1] in ("'", '"'):
        assert _QUOTED.fullmatch(value), "unsupported multi-line quoted scalar"
    return value
def _assert_checkout_credentials(job: str) -> None:
    """Admit only a strict literal step grammar; unsupported syntax fails closed.

    Plain lowercase step keys at exactly 6 and 8 spaces, plain with/env entry
    keys (env may be uppercase) at 10 spaces, no tabs in indentation, no duplicate keys, block scalars
    only as ``|``/``>`` with optional chomping, and every ``uses`` value an
    exact lowercase SHA-pinned reference (optionally in matched quotes
    without escapes). This is a bounded source contract, not a YAML parser
    or a runtime credential attestation.
    """
    lines = job.split("\n")
    starts = [i for i, line in enumerate(lines) if line == "    steps:"]
    assert len(starts) == 1, "unsupported steps syntax"
    steps, step, mapping, scalar_indent = [], None, None, None
    for line in lines[starts[0] + 1:]:
        indent = len(line) - len(line.lstrip(" "))
        if scalar_indent is not None:
            if line.strip(" ") == "" or indent > scalar_indent:
                continue
            scalar_indent = None
        if line.strip(" \t") == "":
            continue
        assert line[indent] != "\t", "tab indentation is unsupported"
        if line[indent] == "#":
            continue
        if indent <= 4:
            job_key = re.fullmatch(r"    ([a-z][a-z0-9_-]*):.*", line)
            assert job_key and job_key.group(1) not in ("uses", "steps", "with"), "unsupported job-level key after steps"
            break
        if match := _STEP.fullmatch(line):
            step, level = {"keys": {}, "with": None}, 6
            steps.append(step)
            mapping = None
        elif step is not None and (match := _STEP_KEY.fullmatch(line)):
            level, mapping = 8, None
        elif mapping is not None and (match := _ENTRY.fullmatch(line)):
            level = 10
        else:
            raise AssertionError("unsupported checkout step syntax")
        key, value = match.group(1), _value(match.group(2))
        target = mapping if level == 10 else step["keys"]
        assert key not in target, "duplicate YAML key"
        target[key] = value
        if re.fullmatch(r"[|>][-+]?", value):
            # A block scalar on a "      - key:" step line belongs to the key
            # column (8), so content must be indented deeper than 8.
            scalar_indent = 8 if level == 6 else level
        elif value == "":
            assert level == 8 and key in ("with", "env"), "unsupported nested value"
            mapping = {}
            if key == "with":
                step["with"] = mapping
        elif key in ("with", "env"):
            raise AssertionError("unsupported flow or alias mapping")
        if key == "uses":
            assert level != 10, "uses is unsupported inside a mapping"
            bare = value[1:-1] if value[:1] in ("'", '"') and value[:1] == value[-1:] else value
            assert _CANONICAL_USES.fullmatch(bare), "non-canonical uses spelling"
    checkouts = [s for s in steps if s["keys"].get("uses", "").strip("'\"").startswith("actions/checkout@")]
    assert len(checkouts) == 1, "expected exactly one checkout per execution job"
    assert set(checkouts[0]["keys"]) <= {"name", "uses", "with"}, "unsupported checkout step keys"
    assert (checkouts[0]["with"] or {}).get("persist-credentials") == "false"


@pytest.mark.parametrize("name", _CHECKOUT_JOBS)
def test_ci_checkouts_do_not_persist_credentials(name: str) -> None:
    """Executable PR code must not inherit a checkout-persisted Git credential."""
    job = _job(_WORKFLOW.read_text(encoding="utf-8"), name)
    _assert_checkout_credentials(job)


def test_ci_keeps_all_existing_execution_job_identities() -> None:
    """Do not remove a failing required producer to satisfy the routing check."""
    workflow = _WORKFLOW.read_text(encoding="utf-8")
    assert tuple(re.findall(r"^  ([A-Za-z0-9_-]+):$", workflow.split("jobs:\n", 1)[1], re.MULTILINE)) == _JOBS


def test_isolated_runner_label_is_declared_without_lint_exemptions() -> None:
    """Use normal actionlint discovery, never an external acceptance override.

    Basis: actionlint v1.7.12 configuration documentation, Configuration file
    and self-hosted-runner.labels sections (read after the initial RED).
    The exact minimal declaration retains every linter check. It establishes
    source configuration only, not eligible capacity or physical isolation.

    References:
        actionlint contributors. (n.d.). Configuration. actionlint
        (Version 1.7.12), Configuration file and self-hosted-runner.labels.
        https://github.com/rhysd/actionlint/blob/v1.7.12/docs/config.md
    """
    config = _ROOT / ".github" / "actionlint.yaml"
    assert config.is_file(), "declare the new isolated runner label for normal lint discovery"
    assert config.read_text(encoding="utf-8") == (
        "# Custom label syntax only; runner capacity and isolation need operator evidence.\n"
        "self-hosted-runner:\n"
        "  labels:\n"
        "    - cwlab-ci-isolated\n"
    ), "declare only the exact custom label; do not add ignores or wildcard exemptions"


# Checkout regression witnesses: source-only variants, no credentials or runner execution.
ACTION='actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1'
VARIANTS=[('plain','', ' ',False),('two_spaces','','  ',False),('tab','','\t',False),
          ('single_quote',"'",' ',False),('double_quote','"',' ',False),('name_first','',' ',True)]
def step(quote,space,name_first,credential):
    """Build one checkout step fixture with the given quoting, spacing and credential flag."""
    action=quote+ACTION+quote
    prefix='      - name: Fixture checkout\n        uses:'+space+action+'\n' if name_first else '      - uses:'+space+action+'\n'
    return prefix+'        with:\n          persist-credentials: '+credential+'\n'
class Text:
    """Stand in for the workflow Path so a test can read transformed workflow text."""
    def __init__(self,text):self.text=text
    def read_text(self,**kwargs):
        """Return the stored workflow text, ignoring encoding arguments."""
        return self.text
@pytest.mark.parametrize('name',_CHECKOUT_JOBS)
@pytest.mark.parametrize('variant,quote,space,name_first',VARIANTS)
@pytest.mark.parametrize('kind',['unsafe_second','safe_singleton'])
def test_checkout_guard_recognizes_literal_step_variants(monkeypatch,name,variant,quote,space,name_first,kind):
    """Quote and spacing variants must neither hide a second checkout nor reject a safe singleton."""
    base=_WORKFLOW.read_text(encoding='utf-8')
    job=_job(base,name)
    matches=list(re.finditer(r'^      - uses: actions/checkout@[^\n]+\n(.*?)(?=^      - |\Z)',job,re.M|re.S))
    assert len(matches)==1
    original=matches[0].group(0)
    assert 'persist-credentials: false' in original
    added=step(quote,space,name_first,'true' if kind=='unsafe_second' else 'false')
    replacement=original+added if kind=='unsafe_second' else added
    changed=job.replace(original,replacement,1)
    assert changed!=job or (kind=='safe_singleton' and variant=='plain')
    start=base.index(job);workflow=base[:start]+changed+base[start+len(job):]
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    # Execute original unchanged test function, not a lookalike parser.
    rejected=False
    try:test_ci_checkouts_do_not_persist_credentials(name)
    except AssertionError:rejected=True
    assert rejected is (kind=='unsafe_second'), (kind,variant,name,'original guard false admission/false rejection')


# Jobs-section witnesses (scratch proposal, cron 2026-10-07 06:04 KST): each
# spelling adds a hosted job or a second runs-on that the literal job slicer
# missed. Source-only; no credentials or runners.
_HOSTED = '    runs-on: ubuntu-latest\n    steps:\n      - run: echo hosted\n'
_JOB_GAPS = {
    'first_double_quoted': '  "evil":\n' + _HOSTED,
    'first_single_quoted': "  'evil':\n" + _HOSTED,
    'first_trailing_comment': '  evil: # hosted\n' + _HOSTED,
    'first_trailing_space': '  evil: \n' + _HOSTED,
    'first_space_before_colon': '  evil :\n' + _HOSTED,
    'first_explicit_key': '  ? evil\n  :\n' + _HOSTED,
    'first_tagged_key': '  !!str evil:\n' + _HOSTED,
    'first_flow_mapping': '  evil: {runs-on: ubuntu-latest, steps: [{run: echo hosted}]}\n',
}
_RUNS_ON_GAPS = {
    'dup_runs_on_quoted': '    "runs-on": ubuntu-latest\n',
    'dup_runs_on_single_quoted': "    'runs-on': ubuntu-latest\n",
    'dup_runs_on_space_colon': '    runs-on : ubuntu-latest\n',
    'dup_runs_on_explicit_key': '    ? runs-on\n    : ubuntu-latest\n',
    'dup_runs_on_trailing_space_block': '    runs-on: \n      - ubuntu-latest\n',
}
def _run_contracts(workflow):
    """Run the routing and job-identity contracts against ``workflow``."""
    for job_name in _JOBS:
        test_ci_jobs_require_isolated_linux_x64_self_hosted_routing(job_name)
    test_ci_keeps_all_existing_execution_job_identities()
@pytest.mark.parametrize('gap', sorted(_JOB_GAPS))
def test_jobs_guard_rejects_unlisted_first_job(monkeypatch, gap):
    """A non-literal first job spelling must not hide an off-pool job."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    workflow = base.replace('\njobs:\n', '\njobs:\n' + _JOB_GAPS[gap], 1)
    assert workflow != base
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    with pytest.raises(AssertionError):
        _run_contracts(workflow)
@pytest.mark.parametrize('name', _JOBS)
@pytest.mark.parametrize('gap', sorted(_RUNS_ON_GAPS))
def test_jobs_guard_rejects_second_runs_on(monkeypatch, name, gap):
    """A non-literal second runs-on must not reroute an existing job."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    job = _job(base, name)
    line = '    runs-on: [self-hosted, Linux, X64, cwlab-ci-isolated]\n'
    assert job.count(line) == 1
    start = base.index(job)
    workflow = base[:start] + job.replace(line, line + _RUNS_ON_GAPS[gap], 1) + base[start + len(job):]
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    with pytest.raises(AssertionError):
        _run_contracts(workflow)
@pytest.mark.parametrize('name', _JOBS)
def test_jobs_guard_admits_plain_key_comment_and_blank_controls(monkeypatch, name):
    """Plain extra job keys, indented comments and blank lines stay admissible."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    job = _job(base, name)
    line = '    runs-on: [self-hosted, Linux, X64, cwlab-ci-isolated]\n'
    start = base.index(job)
    extra = '    continue-on-error: false\n\n  # control comment\n    # control comment\n'
    assert 'continue-on-error' not in job
    workflow = base[:start] + job.replace(line, line + extra, 1) + base[start + len(job):]
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    _run_contracts(workflow)

# Open-gap witnesses (scratch proposal, cron 2026-10-07): each transform keeps the
# real checkout step and adds one spelling that hid a second or credentialed
# checkout from the literal helper. Source-only; no credentials or runners.
_SHA = '3d3c42e5aac5ba805825da76410c181273ba90b1'
_A = 'actions/checkout@' + _SHA
def _second(value):
    """Build a second credentialed checkout step that spells ``uses`` as ``value``."""
    return ('      - name: Second checkout\n        uses: ' + value +
            '\n        with:\n          persist-credentials: true\n')
_GAPS = {
    'dup_key_in_with': lambda o: o.replace('persist-credentials: false\n',
        'persist-credentials: false\n          persist-credentials: true\n', 1),
    'split_indent10': lambda o: o + '          uses: ' + _A +
        '\n          with:\n            persist-credentials: true\n',
    'split_six_space_uses': lambda o: o + '      uses: ' + _A +
        '\n        with:\n          persist-credentials: true\n',
    'second_uses_tab': lambda o: o.replace('        with:\n', '\tuses: ' + _A + '\n        with:\n', 1),
    'encoded_upper': lambda o: o + _second('Actions/Checkout@' + _SHA),
    'encoded_all_upper': lambda o: o + _second('ACTIONS/CHECKOUT@' + _SHA),
    'encoded_dq_slash': lambda o: o + _second('"actions\\/checkout@' + _SHA + '"'),
    'encoded_dq_hex': lambda o: o + _second('"\\x61ctions/checkout@' + _SHA + '"'),
    'encoded_dq_unicode': lambda o: o + _second('"\\u0061ctions/checkout@' + _SHA + '"'),
}
for _k in ('name', 'run', 'if'):
    for _ind in ('|', '>-'):
        _GAPS['step_line_block_%s_%s' % (_k, _ind)] = (lambda o, k=_k, ind=_ind: o +
            '      - ' + k + ': ' + ind + '\n        uses: ' + _A +
            '\n        with:\n          persist-credentials: true\n')
for _n in (4, 6, 7, 9, 10, 12):
    _GAPS['second_uses_indent%d' % _n] = (lambda o, n=_n: o.replace(
        '        with:\n', ' ' * n + 'uses: ' + _A + '\n        with:\n', 1))
@pytest.mark.parametrize('name', _CHECKOUT_JOBS)
@pytest.mark.parametrize('gap', sorted(_GAPS))
def test_checkout_guard_rejects_open_gap_witnesses(monkeypatch, name, gap):
    """Each open-gap spelling of a second or credentialed checkout must fail closed."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    job = _job(base, name)
    matches = list(re.finditer(r'^      - uses: actions/checkout@[^\n]+\n(.*?)(?=^      - |\Z)', job, re.M | re.S))
    assert len(matches) == 1
    original = matches[0].group(0)
    changed = job.replace(original, _GAPS[gap](original), 1)
    assert changed != job
    start = base.index(job)
    workflow = base[:start] + changed + base[start + len(job):]
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    with pytest.raises(AssertionError):
        test_ci_checkouts_do_not_persist_credentials(name)


# Line-break witnesses (scratch proposal, cron 2026-10-07 06:34 KST): NEL, LS
# and PS are line breaks for YAML 1.1 readers (PyYAML, libyaml/Psych, go-yaml
# in actionlint) but not for str.split("\n"), so a comment line can hide a job,
# a second runs-on or a second credentialed checkout. Source-only fixtures.
_BREAKS = {'nel': '\x85', 'ls': '\u2028', 'ps': '\u2029'}
_ISOLATED = '    runs-on: [self-hosted, Linux, X64, cwlab-ci-isolated]\n'
def _all_contracts():
    """Run routing, identity and checkout contracts on the patched workflow."""
    _run_contracts(None)
    for job_name in _CHECKOUT_JOBS:
        test_ci_checkouts_do_not_persist_credentials(job_name)
def _break_cases(base):
    """Yield (case, workflow) pairs that hide a hazard behind a non-LF break."""
    for bname, br in sorted(_BREAKS.items()):
        yield bname + ':first_job', base.replace('\njobs:\n', '\njobs:\n  # note' + br +
            '  evil: {runs-on: ubuntu-latest, steps: [{run: echo hosted}]}\n', 1)
        for name in _JOBS:
            job = _job(base, name)
            start = base.index(job)
            yield bname + ':runs_on:' + name, base[:start] + job.replace(
                _ISOLATED, _ISOLATED + '    # note' + br + '    runs-on: ubuntu-latest\n', 1) + base[start + len(job):]
        for name in _CHECKOUT_JOBS:
            job = _job(base, name)
            start = base.index(job)
            yield bname + ':persist:' + name, base[:start] + job.replace(
                '          persist-credentials: false\n', '          persist-credentials: false\n'
                '          # note' + br + '          persist-credentials: true\n', 1) + base[start + len(job):]
            yield bname + ':flow_checkout:' + name, base[:start] + job.replace(
                '\n    steps:\n', '\n    steps:\n      # note' + br + '      - {uses: ' + _A +
                ', with: {persist-credentials: true}}\n', 1) + base[start + len(job):]
_BREAK_IDS = [case for case, _ in _break_cases(_WORKFLOW.read_text(encoding='utf-8'))]
@pytest.mark.parametrize('case', _BREAK_IDS)
def test_contracts_reject_non_lf_line_breaks(monkeypatch, case):
    """A NEL/LS/PS break inside a comment must not hide a job, route or checkout."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    workflow = dict(_break_cases(base))[case]
    assert workflow != base
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    with pytest.raises(AssertionError):
        _all_contracts()
def test_contracts_admit_non_ascii_comment_control(monkeypatch):
    """Ordinary non-ASCII comment text stays admissible; only breaks/controls fail."""
    base = _WORKFLOW.read_text(encoding='utf-8')
    workflow = base.replace('\njobs:\n', '\njobs:\n  # \uc8fc\uc11d \u00e9 note\n', 1)
    assert workflow != base
    monkeypatch.setattr(__import__(__name__, fromlist=['_WORKFLOW']), '_WORKFLOW', Text(workflow))
    _all_contracts()
