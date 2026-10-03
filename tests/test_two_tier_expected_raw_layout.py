"""Python marshaling regressions; no native numerical/GPU execution.

The plug-in expected raw estimand remains unchanged (Cai, 2015, pp. 542–543,
Eqs. 14–17). Contiguous buffers are a local PyO3 as_slice transport invariant,
not a psychometric assumption or a claim of quadrature accuracy.

Reference:
Cai, L. (2015). Lord–Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535–559.
https://doi.org/10.1007/s11336-014-9411-3
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'python' / 'fast_mlsirm' / 'two_tier_grm.py'


@pytest.fixture
def seam(monkeypatch):
    """Load exact Python source; a stand-in asserts the native buffer contract."""
    namespace = '_personscore_layout_contract'
    package = ModuleType(namespace)
    package.__path__ = [str(SOURCE.parent)]
    calls = []

    def binding(*args):
        arrays = args[:5]
        expected_dtypes = [np.dtype('float64')] * 4 + [np.dtype('int64')]
        for array, dtype in zip(arrays, expected_dtypes):
            assert array.ndim == 1
            assert array.dtype == dtype
            assert array.flags.c_contiguous, 'Rust as_slice requires contiguous arrays'
            assert array.flags.aligned, 'Rust as_slice requires aligned buffers'
        calls.append(args)
        return np.array([1.25, 2.0, 2.75])

    fitstats = ModuleType(namespace + '.fitstats')
    setattr(fitstats, '_core_module', lambda: SimpleNamespace(two_tier_expected_raw=binding))
    module = ModuleType(namespace + '.two_tier_grm')
    module.__file__ = str(SOURCE)
    module.__package__ = namespace
    for item in (package, fitstats, module):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    exec(compile(SOURCE.read_bytes(), str(SOURCE), 'exec'), module.__dict__)
    return module, calls


@pytest.fixture
def fit():
    """Ordinary values with no fitted-estimate or recovered-trait claim."""
    return SimpleNamespace(
        a_primary=np.array([[1.2, 0.3], [0.4, 1.0]]),
        a_specific=np.array([0.5, 0.4]),
        threshold=np.array([[0.8, -0.8], [1.0, -1.0]]),
        theta_p_eap=np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
        n_primary=2, n_specific=1, n_cat=3,
    )


def strided(values):
    """Preserve logical values in a real noncontiguous view, not a fake flag."""
    padded = np.zeros(tuple(2 * n for n in values.shape), dtype=values.dtype)
    view = padded[tuple(slice(None, None, 2) for _ in values.shape)]
    view[...] = values
    assert not view.flags.c_contiguous
    return view


@pytest.mark.parametrize('device', ['cpu', 'auto', 'gpu'])
@pytest.mark.parametrize('field', ['a_primary', 'a_specific', 'threshold', 'theta_p_eap'])
@pytest.mark.parametrize('layout', ['positive-stride', 'negative-stride'])
def test_expected_raw_materializes_views_without_value_changes(seam, fit, field, layout, device):
    """Every native float buffer must be contiguous without reordering values."""
    module, calls = seam
    original = getattr(fit, field)
    view = strided(original)
    if layout == 'negative-stride':
        view = view[::-1]
    setattr(fit, field, view)
    expected = [np.asarray(getattr(fit, name)).reshape(-1).copy()
                for name in ('a_primary', 'a_specific', 'threshold', 'theta_p_eap')]
    actual = module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121, device=device)
    np.testing.assert_array_equal(actual, [1.25, 2.0, 2.75])
    assert len(calls) == 1
    for transported, logical in zip(calls[0][:4], expected):
        np.testing.assert_array_equal(transported, logical)
    assert calls[0][5:] == (3, 2, 1, 121, device)
    np.testing.assert_array_equal(getattr(fit, field), view)
    assert not getattr(fit, field).flags.c_contiguous


@pytest.mark.parametrize('field', ['a_primary', 'threshold', 'theta_p_eap'])
def test_expected_raw_single_column_views_are_contiguous(seam, fit, field):
    """reshape(-1) can preserve a stride when a matrix has one column."""
    module, calls = seam
    fit.n_primary = 1
    fit.n_cat = 2
    fit.a_primary = np.array([[1.2], [1.0]])
    fit.threshold = np.array([[0.8], [1.0]])
    fit.theta_p_eap = np.array([[-1.0], [0.0], [1.0]])
    view = strided(getattr(fit, field))
    setattr(fit, field, view)
    expected = view.reshape(-1).copy()
    module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121)
    index = {'a_primary': 0, 'threshold': 2, 'theta_p_eap': 3}[field]
    np.testing.assert_array_equal(calls[0][index], expected)
    assert calls[0][5:] == (2, 1, 1, 121, 'cpu')


@pytest.mark.parametrize('device', ['cpu', 'auto', 'gpu'])
def test_expected_raw_contiguous_controls_preserve_all_arguments(seam, fit, device):
    """The ordinary positive control reaches one unchanged numerical call."""
    module, calls = seam
    module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 241, device=device)
    assert len(calls) == 1
    assert calls[0][5:] == (3, 2, 1, 241, device)


@pytest.mark.parametrize('field', ['a_primary', 'a_specific', 'threshold', 'theta_p_eap'])
def test_expected_raw_aligns_contiguous_float_buffers(seam, fit, field):
    """A contiguous unaligned view must retain values and meet Rust slice admission."""
    module, calls = seam
    original = getattr(fit, field).copy()
    storage = bytearray(original.nbytes + 1)
    view = np.ndarray(original.shape, dtype=np.float64, buffer=storage, offset=1)
    view[...] = original
    assert view.flags.c_contiguous and not view.flags.aligned
    setattr(fit, field, view)
    expected = [getattr(fit, name).reshape(-1).copy()
                for name in ('a_primary', 'a_specific', 'threshold', 'theta_p_eap')]
    actual = module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121)
    np.testing.assert_array_equal(actual, [1.25, 2.0, 2.75])
    assert len(calls) == 1
    for transported, logical in zip(calls[0][:4], expected):
        np.testing.assert_array_equal(transported, logical)
    assert calls[0][5:] == (3, 2, 1, 121, 'cpu')
    np.testing.assert_array_equal(getattr(fit, field), original)
    assert not getattr(fit, field).flags.aligned


def test_expected_raw_invalid_device_does_not_reach_binding(seam, fit):
    """Layout normalization must not relax an unrelated public admission guard."""
    module, calls = seam
    with pytest.raises(ValueError, match='device'):
        module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121, device='cuda')
    assert calls == []
