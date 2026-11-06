"""Synthetic orchestration tests; no fitting or statistical claims."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'python/fast_mlsirm/sequential_dif.py'
spec = importlib.util.spec_from_file_location('sequential_dif_owned', SOURCE)
assert spec is not None and spec.loader is not None
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def request():
    return dict(item_ids=['a', 'b', 'c'], anchor=[True]*3, device='gpu',
                controls={'require_gpu': True, 'prior': 'retained', 'hessian': 'retained'},
                max_rounds=3, studied_item=None, alpha=.05, y=[0, 1, 0])


class SyntheticNative:
    """Recording contract fixture, deliberately NOT a GPU implementation."""
    def __init__(self):
        self.calls = []

    def sequential_dif_capabilities(self):
        return {'schema_version': 1, 'fit_backend': 'gpu_only',
                'full_matrix_single_removal': True, 'all_items_retested': True,
                'native_identity': 'synthetic-test-only'}

    def sequential_dif_round_gpu(self, raw):
        req = json.loads(raw)
        self.calls.append(req)
        removed = 1 if req['anchor'][1] else None
        final = list(req['anchor'])
        if removed is not None:
            final[removed] = False
        status = 'retest_all' if removed is not None else 'stable_no_further_removal'
        rows = [dict(tested_item=i, status='complete', p_value=.01 if i == 1 else .8)
                for i in range(3)]
        return json.dumps(dict(request_sha256=controller._digest(raw),
            fit_receipt={'backend': 'gpu_only', 'e_step_gpu_dispatches': 1,
                         'm_step_gpu_dispatches': 1, 'cpu_fit_dispatches': 0},
            result=dict(rounds=[dict(anchors_before=req['anchor'], item_results=rows,
                        adjusted_p=[.03,.03,1.], removed=removed, status=status)],
                        final_anchor=final,
                        status='round_limit_incomplete' if removed is not None else status)))


class ControllerTests(unittest.TestCase):
    def test_automatic_full_retest_preserves_settings(self):
        backend = SyntheticNative()
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            result = controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(result['status'], 'stable_no_further_removal')
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(backend.calls[1]['anchor'], [True, False, True])
            self.assertEqual([len(r['item_results']) for r in result['rounds']], [3, 3])
            for call in backend.calls:
                self.assertEqual(call['controls'], request()['controls'])
                self.assertEqual(call['y'], request()['y'])
                self.assertEqual(call['max_rounds'], 1)
                self.assertIsNone(call['studied_item'])


    def test_cpu_or_missing_native_contract_refused_before_fit(self):
        for backend in (object(), SyntheticNative()):
            if isinstance(backend, SyntheticNative):
                backend.sequential_dif_capabilities = lambda: {'fit_backend': 'cpu'}
            with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
                with self.assertRaisesRegex(RuntimeError, 'GPU'):
                    controller.run_sequential_dif(request(), archive=root)
                if isinstance(backend, SyntheticNative):
                    self.assertEqual(backend.calls, [])

    def test_cpu_request_refused(self):
        req = request()
        req['device'] = 'cpu'
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=SyntheticNative()):
            with self.assertRaises(ValueError):
                controller.run_sequential_dif(req, archive=root)

    def test_cpu_mstep_receipt_stops_and_preserves_result(self):
        backend = SyntheticNative()
        native = backend.sequential_dif_round_gpu
        def wrong(raw):
            output = json.loads(native(raw))
            output['fit_receipt']['m_step_gpu_dispatches'] = 0
            return json.dumps(output)
        backend.sequential_dif_round_gpu = wrong
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            result = controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(result['status'], 'native_contract_failed')
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(result['final_anchor'], [True]*3)
            self.assertTrue(list(Path(root).glob('round-*.json')))

    def test_resume_reuses_complete_rounds_and_rejects_drift(self):
        backend = SyntheticNative()
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            first = controller.run_sequential_dif(request(), archive=root)
            second = controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(first, second)
            self.assertEqual(len(backend.calls), 2)
            req = request()
            req['controls']['prior'] = 'changed'
            with self.assertRaises(ValueError):
                controller.run_sequential_dif(req, archive=root)

    def test_failed_family_is_retained_without_removal(self):
        backend = SyntheticNative()
        native = backend.sequential_dif_round_gpu
        def failed(raw):
            out = json.loads(native(raw))
            r = out['result']['rounds'][0]
            r.update(adjusted_p=None, removed=None, status='round_incomplete')
            r['item_results'][1].update(status='pair_failed', p_value=None,
                error='item b group 1 category 3 unobserved; boundary unsupported')
            out['result'].update(final_anchor=[True]*3, status='round_incomplete')
            return json.dumps(out)
        backend.sequential_dif_round_gpu = failed
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            result = controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(result['status'], 'round_incomplete')
            self.assertEqual(len(result['rounds'][0]['item_results']), 3)
            self.assertEqual(result['rounds'][0]['item_results'][1]['error'],
                'item b group 1 category 3 unobserved; boundary unsupported')
            self.assertEqual(backend.calls[0]['controls'], request()['controls'])
            self.assertEqual(backend.calls[0]['y'], request()['y'])
            self.assertEqual(len(backend.calls), 1)

    def test_pending_round_does_not_automatically_refit(self):
        backend = SyntheticNative()
        backend.sequential_dif_round_gpu = lambda raw: (_ for _ in ()).throw(KeyboardInterrupt())
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            with self.assertRaises(KeyboardInterrupt):
                controller.run_sequential_dif(request(), archive=root)
            with self.assertRaisesRegex(RuntimeError, 'unsettled'):
                controller.run_sequential_dif(request(), archive=root)

    def test_round_cap_reports_incomplete(self):
        req = request()
        req['max_rounds'] = 1
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=SyntheticNative()):
            result = controller.run_sequential_dif(req, archive=root)
            self.assertEqual(result['status'], 'round_limit_incomplete')

    def test_output_contract_negative_controls(self):
        mutations = [
            lambda o: o.update(request_sha256='wrong'),
            lambda o: o['fit_receipt'].update(e_step_gpu_dispatches=0),
            lambda o: o['fit_receipt'].update(cpu_fit_dispatches=1),
            lambda o: o['result']['rounds'][0]['item_results'].pop(),
            lambda o: o['result']['rounds'][0].update(removed=99),
            lambda o: o['result'].update(final_anchor=[False]*3),
            lambda o: o['result']['rounds'][0].update(adjusted_p=[.1]),
            lambda o: o['result']['rounds'][0]['item_results'][0].update(status='pair_failed'),
        ]
        for mutate in mutations:
            backend = SyntheticNative()
            native = backend.sequential_dif_round_gpu
            def changed(raw):
                out = json.loads(native(raw))
                mutate(out)
                return json.dumps(out)
            backend.sequential_dif_round_gpu = changed
            with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
                result = controller.run_sequential_dif(request(), archive=root)
                self.assertEqual(result['status'], 'native_contract_failed')
                self.assertEqual(len(backend.calls), 1)

    def test_native_identity_types_are_strict(self):
        for field in ('tested_item', 'anchors_before', 'final_anchor'):
            with self.subTest(field=field):
                backend = SyntheticNative()
                native = backend.sequential_dif_round_gpu
                def changed(raw):
                    output = json.loads(native(raw))
                    row = output['result']['rounds'][0]
                    if field == 'tested_item':
                        for item, value in zip(row['item_results'], [False, True, 2.0]):
                            item['tested_item'] = value
                    elif field == 'anchors_before':
                        row[field] = [int(a) for a in row[field]]
                    else:
                        output['result'][field] = [int(a) for a in output['result'][field]]
                    return json.dumps(output)
                backend.sequential_dif_round_gpu = changed
                with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
                    result = controller.run_sequential_dif(request(), archive=root)
                    self.assertEqual(result['status'], 'native_contract_failed')
                    self.assertEqual(result['final_anchor'], [True]*3)
                    self.assertEqual(len(backend.calls), 1)

    def test_replay_identity_types_are_strict_without_refitting(self):
        for field in ('tested_item', 'anchors_before', 'final_anchor'):
            with self.subTest(field=field):
                backend = SyntheticNative()
                with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
                    controller.run_sequential_dif(request(), archive=root)
                    path = Path(root) / 'round-0001.json'
                    packet = json.loads(path.read_text())
                    output = packet['output']
                    row = output['result']['rounds'][0]
                    if field == 'tested_item':
                        for item, value in zip(row['item_results'], [False, True, 2.0]):
                            item['tested_item'] = value
                    elif field == 'anchors_before':
                        row[field] = [int(a) for a in row[field]]
                    else:
                        output['result'][field] = [int(a) for a in output['result'][field]]
                    packet['sha256'] = controller._digest(controller._json(output))
                    path.write_text(json.dumps(packet))
                    result = controller.run_sequential_dif(request(), archive=root)
                    self.assertEqual(result['status'], 'native_contract_failed')
                    self.assertEqual(result['final_anchor'], [True]*3)
                    self.assertEqual(len(backend.calls), 2)

    def test_last_anchor_removal_is_rejected_fresh_and_replayed(self):
        for max_rounds in (1, 3):
            for replay in (False, True):
                with self.subTest(max_rounds=max_rounds, replay=replay):
                    req = request()
                    req.update(anchor=[True, False, False], max_rounds=max_rounds)
                    backend = SyntheticNative()
                    native = backend.sequential_dif_round_gpu

                    def remove_last(raw):
                        output = json.loads(native(raw))
                        row = output['result']['rounds'][0]
                        row.update(removed=0, status='retest_all')
                        output['result']['final_anchor'] = [False, False, False]
                        return json.dumps(output)

                    backend.sequential_dif_round_gpu = remove_last
                    with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
                        result = controller.run_sequential_dif(req, archive=root)
                        if replay:
                            result = controller.run_sequential_dif(req, archive=root)
                        self.assertEqual(result['status'], 'native_contract_failed')
                        self.assertEqual(result['final_anchor'], req['anchor'])
                        self.assertEqual(len(backend.calls), 1)
                        self.assertEqual(result['rounds'], [])

    def test_archive_mutation_is_rejected_without_fitting(self):
        backend = SyntheticNative()
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            controller.run_sequential_dif(request(), archive=root)
            path = Path(root) / 'round-0001.json'
            packet = json.loads(path.read_text())
            packet['output']['request_sha256'] = 'mutant'
            path.write_text(json.dumps(packet))
            with self.assertRaises(ValueError):
                controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(len(backend.calls), 2)

    def test_same_archive_lock_prevents_dispatch(self):
        import fcntl
        backend = SyntheticNative()
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            with (Path(root) / '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(backend.calls, [])

    def test_exception_preserved_and_not_retried(self):
        backend = SyntheticNative()
        backend.sequential_dif_round_gpu = lambda raw: (_ for _ in ()).throw(RuntimeError('synthetic failure'))
        with tempfile.TemporaryDirectory() as root, patch.object(controller, '_native', return_value=backend):
            result = controller.run_sequential_dif(request(), archive=root)
            self.assertEqual(result['failed_output']['native_error'], 'synthetic failure')
            self.assertEqual(result, controller.run_sequential_dif(request(), archive=root))

if __name__ == '__main__':
    unittest.main()
