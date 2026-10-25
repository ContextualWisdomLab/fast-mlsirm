"""Automatic full-matrix DIF orchestration; all statistics stay native.

The one-removal policy follows the retained native ``purify_full_matrix``
producer, not mirt's batch constraint updates. See docs/sequential_dif.md.
This module does not implement a fitter, a p adjustment, or a Hessian.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import fcntl
from pathlib import Path

__all__ = ['run_sequential_dif']


def _digest(raw):
    return hashlib.sha256(raw.encode()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _native():
    return importlib.import_module('fast_mlsirm._core')


def _round(backend, request):
    return json.loads(backend.sequential_dif_round_gpu(_json(request)))


def run_sequential_dif(request, *, archive):
    """Run bounded automatic rounds through the strict native GPU contract.

    Exposed as ``fast_mlsirm.sequential_dif.run_sequential_dif``. No caller
    callback can replace the native fitter. The current main native library
    lacks this strict E+M entry point, and is refused before fitting. See the
    companion contract for the independent GPU M-step owner's integration.
    Native exceptions are archived as failed output without automatic retry;
    they are not re-raised as their original Python exception class.

    Chalmers (2012, Section 3.1, pp. 5-7) describes the EM estimation context.
    The single-removal and archive policies are software contracts, not
    procedures validated by that source.

    References
    ----------
    Chalmers, R. P. (2012). mirt: A multidimensional item response theory
    package for the R environment. Journal of Statistical Software, 48(6),
    1-29. https://doi.org/10.18637/jss.v048.i06
    """
    request = json.loads(_json(request))
    anchors = request.get('anchor')
    ids = request.get('item_ids')
    if (request.get('device') != 'gpu'
            or request.get('controls', {}).get('require_gpu') is not True
            or request.get('studied_item') is not None
            or type(request.get('max_rounds')) is not int or request['max_rounds'] < 1
            or not isinstance(anchors, list) or not anchors
            or any(type(a) is not bool for a in anchors) or not any(anchors)
            or not isinstance(ids, list) or len(ids) != len(anchors)
            or any(not isinstance(i, str) or not i for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('invalid full-family GPU request')
    backend = _native()
    capability = getattr(backend, 'sequential_dif_capabilities', lambda: {})()
    if (capability.get('schema_version') != 1
            or capability.get('fit_backend') != 'gpu_only'
            or capability.get('full_matrix_single_removal') is not True
            or capability.get('all_items_retested') is not True
            or not capability.get('native_identity')
            or not callable(getattr(backend, 'sequential_dif_round_gpu', None))):
        raise RuntimeError('native strict GPU E+M contract unavailable; no fit dispatched')
    root = Path(archive)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(schema_version=1, request_sha256=_digest(_json(request)),
                        capability=capability)
        manifest = root / 'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('archive request/native identity mismatch')
        else:
            _save(manifest, identity)
        rounds = []
        status = 'round_limit_incomplete'
        for index in range(request['max_rounds']):
            current = dict(request, anchor=anchors[:], max_rounds=1, studied_item=None)
            path = root / f'round-{index + 1:04d}.json'
            pending = root / f'round-{index + 1:04d}.pending'
            if path.exists():
                packet = json.loads(path.read_text())
                if packet['sha256'] != _digest(_json(packet['output'])):
                    raise ValueError('archive round digest mismatch')
                output = packet['output']
            else:
                if pending.exists():
                    raise RuntimeError('unsettled native round; explicit owner reconciliation required')
                _save(pending, {'request_sha256': _digest(_json(current))})
                try:
                    output = _round(backend, current)
                except Exception as error:
                    output = {'native_error': str(error)}
                _save(path, {'output': output, 'sha256': _digest(_json(output))})
            try:
                row, after = _validate_output(output, current)
            except (ValueError, KeyError, TypeError, IndexError):
                return dict(status='native_contract_failed', rounds=rounds,
                            final_anchor=anchors, failed_output=output)
            rounds.append(row)
            anchors = after
            status = row['status']
            if status != 'retest_all':
                break
            status = 'round_limit_incomplete'
        return dict(status=status, rounds=rounds, final_anchor=anchors)


def _save(path, value):
    # Exclusive publication: partial writes remain pending and cannot authorize replay.
    with path.open('x') as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())


def _validate_output(output, request):
    if output['request_sha256'] != _digest(_json(request)):
        raise ValueError('native request drift')
    receipt = output['fit_receipt']
    if (receipt['backend'] != 'gpu_only'
            or type(receipt['e_step_gpu_dispatches']) is not int
            or receipt['e_step_gpu_dispatches'] < 1
            or type(receipt['m_step_gpu_dispatches']) is not int
            or receipt['m_step_gpu_dispatches'] < 1
            or type(receipt['cpu_fit_dispatches']) is not int
            or receipt['cpu_fit_dispatches'] != 0):
        raise ValueError('GPU E/M receipt incomplete')
    result = output['result']
    if len(result['rounds']) != 1:
        raise ValueError('one native round required')
    row = result['rounds'][0]
    anchors = request['anchor']
    n = len(anchors)
    inventory = row['item_results']
    before = row['anchors_before']
    if (type(before) is not list or any(type(a) is not bool for a in before)
            or before != anchors or type(inventory) is not list
            or any(type(r['tested_item']) is not int for r in inventory)
            or [r['tested_item'] for r in inventory] != list(range(n))):
        raise ValueError('incomplete or reordered family')
    removed = row['removed']
    after = anchors[:]
    complete = all(r['status'] == 'complete' for r in row['item_results'])
    if not complete:
        if removed is not None or row['adjusted_p'] is not None or row['status'] not in ('round_incomplete', 'identification_exhausted'):
            raise ValueError('failed family cannot remove anchors')
    else:
        adjusted = row['adjusted_p']
        if not isinstance(adjusted, list) or len(adjusted) != n:
            raise ValueError('native adjustment family incomplete')
        for p in adjusted + [r['p_value'] for r in row['item_results']]:
            if type(p) not in (int, float) or not 0 <= p <= 1:
                raise ValueError('invalid native p value')
        if removed is not None:
            if type(removed) is not int or not 0 <= removed < n or not anchors[removed] or row['status'] != 'retest_all':
                raise ValueError('invalid single removal')
            after[removed] = False
            if not any(after):
                raise ValueError('cannot remove the final anchor')
        elif row['status'] != 'stable_no_further_removal':
            raise ValueError('invalid terminal status')
    final = result['final_anchor']
    if (type(final) is not list or any(type(a) is not bool for a in final)
            or final != after):
        raise ValueError('anchor transition mismatch')
    return row, after
