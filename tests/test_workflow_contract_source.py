"""Reject producer substitutions while resolving immutable central contracts."""
from pathlib import Path
import subprocess
from unittest.mock import Mock

import pytest

from tests.workflow_contract_source import workflow_source


def test_producer_reader_uses_declared_git_object_not_dirty_tree(tmp_path, monkeypatch):
    repository = tmp_path / 'central'
    repository.mkdir()
    subprocess.run(['git', 'init', '-q', str(repository)], check=True)
    source = repository / '.github/workflows/fast-mlsirm-ci.yml'
    source.parent.mkdir(parents=True)
    source.write_text('on:\n  workflow_call:\njobs: {}\n')
    subprocess.run(['git', '-C', str(repository), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(repository), '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'producer'], check=True)
    sha = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    caller = tmp_path / 'ci.yml'
    caller.write_text('jobs:\n  central:\n    uses: ContextualWisdomLab/.github/.github/workflows/fast-mlsirm-ci.yml@' + sha + '\n')
    source.write_text('untrusted dirty producer')
    monkeypatch.setenv('CWL_CENTRAL_CONTRACT_REPOSITORY', str(repository))
    assert workflow_source(caller) == 'on:\n  workflow_call:\njobs: {}\n'


@pytest.mark.parametrize('reference', ['main', 'a'*39, 'a'*41])
def test_mutable_or_malformed_producer_ref_is_rejected_before_network(tmp_path, monkeypatch, reference):
    caller = tmp_path / 'ci.yml'
    caller.write_text('jobs:\n  central:\n    uses: ContextualWisdomLab/.github/.github/workflows/fast-mlsirm-ci.yml@' + reference + '\n')
    fetch = Mock(side_effect=AssertionError('Unexpected network call'))
    monkeypatch.setattr('urllib.request.urlopen', fetch)
    with pytest.raises(AssertionError, match='immutable SHA'):
        workflow_source(caller)
    fetch.assert_not_called()


def test_different_producer_path_cannot_replace_contract(tmp_path):
    caller = tmp_path / 'ci.yml'
    caller.write_text('jobs:\n  central:\n    uses: ContextualWisdomLab/.github/.github/workflows/fast-mlsirm-publish-pypi.yml@' + 'a'*40 + '\n')
    with pytest.raises(AssertionError, match='Unexpected central workflow path'):
        workflow_source(caller)


def test_cached_producer_does_not_hide_mutated_caller(tmp_path, monkeypatch):
    caller = tmp_path / 'ci.yml'
    caller.write_text('local definition')
    assert workflow_source(caller) == 'local definition'
    caller.write_text('jobs:\n  central:\n    uses: ContextualWisdomLab/.github/.github/workflows/fast-mlsirm-ci.yml@main\n')
    with pytest.raises(AssertionError, match='immutable SHA'):
        workflow_source(caller)


def test_network_producer_is_bounded_immutable_and_reusable(tmp_path, monkeypatch):
    import io
    from tests.workflow_contract_source import _producer_source

    sha = "b" * 40
    url = "https://raw.githubusercontent.com/ContextualWisdomLab/.github/" + sha + "/.github/workflows/fast-mlsirm-ci.yml"
    caller = tmp_path / "ci.yml"
    caller.write_text("jobs:\n  central:\n    uses: ContextualWisdomLab/.github/.github/workflows/fast-mlsirm-ci.yml@" + sha + "\n")
    monkeypatch.delenv("CWL_CENTRAL_CONTRACT_REPOSITORY", raising=False)
    for source, destination, error in [
        (b"on:\n  workflow_call:\n", url, None),
        (b"on:\n  workflow_call:\n", url + "-redirect", "redirect"),
        (b"x" * 2_000_001, url, "oversized"),
        (b"on:\n  push:\n", url, "reusable"),
    ]:
        _producer_source.cache_clear()
        response = io.BytesIO(source)
        response.url = destination
        fetch = Mock(return_value=response)
        monkeypatch.setattr("urllib.request.urlopen", fetch)
        if error:
            with pytest.raises(AssertionError, match=error):
                workflow_source(caller)
        else:
            assert workflow_source(caller) == source.decode()
        fetch.assert_called_once_with(url, timeout=30)
    _producer_source.cache_clear()
