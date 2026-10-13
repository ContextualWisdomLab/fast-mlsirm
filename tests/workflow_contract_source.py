"""Read the declared immutable producer for local or centralized workflow tests."""
from functools import cache
import os
from pathlib import Path
import re
import subprocess
import urllib.request

_PREFIX = 'ContextualWisdomLab/.github/.github/workflows/'
_CALL = re.compile(r'(?m)^\s+uses: ' + re.escape(_PREFIX) + r'(fast-mlsirm-[a-z0-9_-]+\.yml)@([0-9a-f]{40})\s*$')


def workflow_source(path: Path) -> str:
    """Resolve exactly one pinned central call, otherwise retain local source.

    An optional central git checkout provides the same immutable object offline;
    ordinary runs fetch the public producer by the full declared commit. Mutable
    references, mixed calls, redirects, oversized content and dirty cache files
    cannot substitute a different producer for a tested contract.
    """
    caller = path.read_text(encoding='utf-8')
    calls = _CALL.findall(caller)
    if not calls:
        if _PREFIX in caller:
            raise AssertionError('Central workflow contract requires a full immutable SHA')
        return caller
    assert len(calls) == caller.count(_PREFIX) == 1, 'One central producer per wrapper is required'
    name, sha = calls[0]
    assert name == 'fast-mlsirm-' + path.name, 'Unexpected central workflow path'
    source = _producer_source(name, sha, os.environ.get('CWL_CENTRAL_CONTRACT_REPOSITORY'))
    assert source and len(source) <= 2_000_000, 'Missing or oversized producer'
    text = source.decode('utf-8')
    assert re.search(r'(?m)^  workflow_call:', text), 'Producer must be reusable'
    return text


@cache
def _producer_source(name: str, sha: str, checkout: str | None) -> bytes:
    """Cache immutable object bytes while callers are revalidated every time."""
    relative = '.github/workflows/' + name
    if checkout:
        # git show reads the commit object, never the potentially dirty worktree.
        result = subprocess.run(['git', '-C', checkout, 'show', sha + ':' + relative],
                                check=True, capture_output=True)
        source = result.stdout
    else:
        url = 'https://raw.githubusercontent.com/ContextualWisdomLab/.github/' + sha + '/' + relative
        with urllib.request.urlopen(url, timeout=30) as response:
            assert response.url == url, 'Producer redirect is not trusted'
            source = response.read(2_000_001)
    return source
