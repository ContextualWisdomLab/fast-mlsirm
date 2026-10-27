"""Synthetic transport checks; no study data or full-GPU claim."""
import json
import os
import subprocess
import sys
from fast_mlsirm import _core


def test_complete_export_rejects_cpu():
    try:
        _core.sequential_dif_complete_gpu('{"device":"cpu"}')
    except ValueError:
        pass
    else:
        raise AssertionError('CPU request admitted')


def test_private_roundtrip_fresh_process(tmp_path):
    # CPU synthetic fixture only; checks persistence without optimizer replay.
    request = dict(y=[0,0,1,1,0,1,1,0]*2, group_id=[0]*4+[1]*4,
                   n_groups=2, n_cat=2, item_ids=['a','b'], anchor=[True,True],
                   q_general=7, max_iter=1, tol=1e-8, n_starts=1, seed=42,
                   newton_iter=2, ridge=1e-6, device='cpu', alpha=0.05, max_rounds=1,
                   controls=dict(fd_step=.001, rank_atol=1e-9, rank_rtol=1e-7,
                       objective_call_budget=10000, workspace_bytes=10000000,
                       ll_roundoff_tolerance=1e-7, fit_ll_tolerance=1e-4,
                       lr_backend_tolerance=1e-4, hessian_rtol=.01, require_gpu=False))
    raw = _core.sequential_dif_fit_private(json.dumps(request))
    path = tmp_path / 'private.json'
    path.write_text(raw)
    path.chmod(0o600)
    from fast_mlsirm import native_lifecycle
    assert native_lifecycle.roundtrip_private(raw) == raw
    before = json.loads(native_lifecycle.restore_summary(raw))
    script = ('import pathlib,sys; from fast_mlsirm import _core; '
              'raw=pathlib.Path(sys.argv[1]).read_text(); '
              'assert _core.sequential_dif_roundtrip_private(raw)==raw; '
              'print(_core.sequential_dif_restore_summary(raw))')
    result = subprocess.run([sys.executable, '-c', script, str(path)],
                            text=True, capture_output=True, timeout=30, check=True)
    assert json.loads(result.stdout) == before
    assert before['converged'] is False
    altered = json.loads(raw)
    state = json.loads(altered['state'])
    state['n_items'] = 99
    altered['state'] = json.dumps(state)
    try:
        _core.sequential_dif_restore_summary(json.dumps(altered))
    except ValueError:
        pass
    else:
        raise AssertionError('corrupt state accepted')
    assert _core.sequential_dif_capabilities()['full_gpu'] is False
