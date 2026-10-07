"""Exact caller-node transport only; never allocate quadrature or execute native code.

The existing recording seam tests the Python-to-binding contract, not a model,
fit, accuracy, device dispatch or public-package initializer claim.
"""
import numpy as np
import pytest

from test_two_tier_expected_raw_layout import seam, fit


@pytest.mark.parametrize('device', ['cpu', 'gpu', 'auto'])
@pytest.mark.parametrize('q', [121, 241, 481, 2**53 - 1, 2**53, 2**53 + 1,
                              2**53 + 3, 2**64 - 1])
def test_node_integer_is_transport_exact(seam, fit, device, q):
    """Recording transport must not round an integer through binary64."""
    module, calls = seam
    original = {name: getattr(fit, name).copy() for name in
                ('a_primary', 'a_specific', 'threshold', 'theta_p_eap')}
    actual = module.expected_raw_two_tier_grm(fit, np.array([0, 0]), q, device=device)
    assert len(calls) == 1
    assert calls[0][5:] == (3, 2, 1, q, device)
    assert type(calls[0][8]) is int
    np.testing.assert_array_equal(actual, [1.25, 2., 2.75])
    for name, before in original.items():
        np.testing.assert_array_equal(getattr(fit, name), before)


SCALARS = list(dict.fromkeys(getattr(np, name) for name in
    ('int8', 'int16', 'int32', 'int64', 'intp', 'longlong',
     'uint8', 'uint16', 'uint32', 'uint64', 'uintp', 'ulonglong')))


@pytest.mark.parametrize('scalar', SCALARS, ids=lambda scalar: scalar.__name__)
def test_concrete_numpy_integer_preserved(seam, fit, scalar):
    """All concrete integer types retain ordinary compatible controls."""
    module, calls = seam
    module.expected_raw_two_tier_grm(fit, np.array([0, 0]), scalar(121))
    assert len(calls) == 1 and calls[0][5:] == (3, 2, 1, 121, 'cpu')


@pytest.mark.parametrize('scalar', [np.int64, np.longlong, np.uint64, np.ulonglong])
def test_large_numpy_integer_preserved(seam, fit, scalar):
    """Signed and unsigned exact inputs must not lose their low bit."""
    module, calls = seam
    q = 2**53 + 1
    module.expected_raw_two_tier_grm(fit, np.array([0, 0]), scalar(q))
    assert len(calls) == 1 and calls[0][8] == q and type(calls[0][8]) is int


@pytest.mark.parametrize('value', [True, 0, -1, 121.5, np.nan, np.inf])
def test_existing_invalid_controls_still_rejected(seam, fit, value):
    """Retain existing errors before the recording binding."""
    module, calls = seam
    with pytest.raises(ValueError, match='q_specific'):
        module.expected_raw_two_tier_grm(fit, np.array([0, 0]), value)
    assert calls == []


def test_integer_subclass_keeps_existing_float_protocol(seam, fit):
    """Do not broaden the proposal into unrelated callback-policy changes."""
    module, calls = seam
    observed = []

    class ExistingProtocol(int):
        def __float__(self):
            observed.append('float')
            return 241.

        def __int__(self):
            raise AssertionError('new callback protocol must not be introduced')

    module.expected_raw_two_tier_grm(fit, np.array([0, 0]), ExistingProtocol(121))
    assert observed == ['float'] and calls[0][8] == 241
