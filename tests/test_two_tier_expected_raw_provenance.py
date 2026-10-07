"""Device provenance for two-tier expected raw scoring; no native execution.

``expected_raw_two_tier_grm`` returns only values, so a GPU request cannot be
distinguished from a silent CPU fallback. ``score_two_tier_grm_expected_raw``
returns the same values together with the Rust dispatcher's ``used_gpu`` flag.
The estimand is unchanged (Cai, 2015, pp. 542-543, Eqs. 14-17); these tests
check transport and admission of the flag, not numerical accuracy.

Reference:
Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
factor models with applications in test scoring, scale alignment, and model
fit testing. Psychometrika, 80(2), 535-559.
https://doi.org/10.1007/s11336-014-9411-3
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import os
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "python" / "fast_mlsirm" / "two_tier_grm.py"
VALUES = np.array([1.25, 2.0, 2.75])


def _load(monkeypatch, core):
    namespace = "_personscore_provenance_contract"
    package = ModuleType(namespace)
    package.__path__ = [str(SOURCE.parent)]
    fitstats = ModuleType(namespace + ".fitstats")
    setattr(fitstats, "_core_module", lambda: core)
    module = ModuleType(namespace + ".two_tier_grm")
    module.__file__ = str(SOURCE)
    module.__package__ = namespace
    for item in (package, fitstats, module):
        monkeypatch.setitem(sys.modules, item.__name__, item)
    exec(compile(SOURCE.read_bytes(), str(SOURCE), "exec"), module.__dict__)
    return module


@pytest.fixture
def fit():
    """Ordinary values; no fitted-estimate or recovered-trait claim."""
    return SimpleNamespace(
        a_primary=np.array([[1.2, 0.3], [0.4, 1.0]]),
        a_specific=np.array([0.5, 0.4]),
        threshold=np.array([[0.8, -0.8], [1.0, -1.0]]),
        theta_p_eap=np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
        n_primary=2, n_specific=1, n_cat=3,
    )


def _recording_core(used_gpu, values=VALUES):
    calls = {"values": [], "provenance": []}

    def plain(*args):
        calls["values"].append(args)
        return values.copy()

    def provenance(*args):
        calls["provenance"].append(args)
        return {"values": values.copy(), "used_gpu": used_gpu}

    core = SimpleNamespace(two_tier_expected_raw=plain,
                           two_tier_expected_raw_with_provenance=provenance)
    return core, calls


@pytest.mark.parametrize("device,used_gpu", [("cpu", False), ("gpu", True), ("gpu", False),
                                            ("auto", True), ("auto", False)])
def test_score_returns_values_and_native_dispatch_flag(monkeypatch, fit, device, used_gpu):
    core, calls = _recording_core(used_gpu)
    module = _load(monkeypatch, core)
    smap = np.array([0, 0])
    result = module.score_two_tier_grm_expected_raw(fit, smap, 121, device=device)
    assert isinstance(result, module.TwoTierGrmExpectedRawScores)
    np.testing.assert_array_equal(result.expected_raw, VALUES)
    assert result.device_requested == device
    assert result.used_gpu is used_gpu
    assert calls["values"] == [] and len(calls["provenance"]) == 1
    args = calls["provenance"][0]
    assert args[5:] == (3, 2, 1, 121, device)


def test_score_marshals_exactly_like_values_only_function(monkeypatch, fit):
    core, calls = _recording_core(False)
    module = _load(monkeypatch, core)
    smap = np.array([0, -1])
    module.expected_raw_two_tier_grm(fit, smap, 121, device="auto")
    module.score_two_tier_grm_expected_raw(fit, smap, 121, device="auto")
    plain, prov = calls["values"][0], calls["provenance"][0]
    assert len(plain) == len(prov) == 10
    for a, b in zip(plain, prov):
        if isinstance(a, np.ndarray):
            assert a.dtype == b.dtype and a.flags.c_contiguous and b.flags.c_contiguous
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b and type(a) is type(b)


def test_values_only_function_does_not_call_provenance_binding(monkeypatch, fit):
    core, calls = _recording_core(True)
    module = _load(monkeypatch, core)
    out = module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121, device="gpu")
    assert type(out) is np.ndarray and calls["provenance"] == []


@pytest.mark.parametrize("payload", [
    {"values": VALUES, "used_gpu": 1},
    {"values": VALUES, "used_gpu": np.bool_(True)},
    {"values": VALUES, "used_gpu": None},
    {"values": VALUES},
    {"used_gpu": False},
    {"values": VALUES, "used_gpu": False, "extra": 0},
    (VALUES, False),
])
def test_malformed_native_provenance_is_rejected(monkeypatch, fit, payload):
    core = SimpleNamespace(two_tier_expected_raw_with_provenance=lambda *a: payload)
    module = _load(monkeypatch, core)
    with pytest.raises(RuntimeError, match="provenance"):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, device="gpu")


def test_cpu_request_cannot_report_gpu_execution(monkeypatch, fit):
    core, _ = _recording_core(True)
    module = _load(monkeypatch, core)
    with pytest.raises(RuntimeError, match="provenance"):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, device="cpu")


@pytest.mark.parametrize("values", [np.array([1.0, np.nan, 2.0]), np.array([1.0, 2.0])])
def test_score_applies_same_output_checks(monkeypatch, fit, values):
    core, _ = _recording_core(False, values=values)
    module = _load(monkeypatch, core)
    with pytest.raises(ValueError, match="expected raw scores"):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, device="cpu")


def test_missing_provenance_binding_fails_closed(monkeypatch, fit):
    core = SimpleNamespace(two_tier_expected_raw=lambda *a: VALUES.copy())
    module = _load(monkeypatch, core)
    with pytest.raises(RuntimeError, match="compiled Rust core"):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, device="cpu")


@pytest.mark.parametrize("bad", [{"device": "metal"}, {"device": True}])
def test_score_validates_device_before_native(monkeypatch, fit, bad):
    core, calls = _recording_core(False)
    module = _load(monkeypatch, core)
    with pytest.raises(ValueError, match="device"):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, **bad)
    assert calls["provenance"] == []


def test_q_specific_remains_required(monkeypatch, fit):
    core, _ = _recording_core(False)
    module = _load(monkeypatch, core)
    with pytest.raises(TypeError):
        module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), device="cpu")


@pytest.mark.parametrize("q", [0, -1])
@pytest.mark.parametrize("entry", ["score_two_tier_grm_expected_raw", "expected_raw_two_tier_grm"])
def test_nonpositive_q_specific_rejected_before_native_on_both_paths(monkeypatch, fit, entry, q):
    core, calls = _recording_core(False)
    module = _load(monkeypatch, core)
    with pytest.raises(ValueError, match="q_specific must be >= 1"):
        getattr(module, entry)(fit, np.array([0, 0]), q, device="cpu")
    assert calls == {"values": [], "provenance": []}


@pytest.mark.parametrize("bad", [{"device": "metal"}, {"device": True}])
def test_legacy_path_shares_device_validation(monkeypatch, fit, bad):
    core, calls = _recording_core(False)
    module = _load(monkeypatch, core)
    with pytest.raises(ValueError, match="device"):
        module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121, **bad)
    assert calls == {"values": [], "provenance": []}


def test_caller_arrays_unchanged(monkeypatch, fit):
    core, _ = _recording_core(True)
    module = _load(monkeypatch, core)
    before = {k: v.copy() for k, v in vars(fit).items() if isinstance(v, np.ndarray)}
    smap = np.array([0, 0])
    module.score_two_tier_grm_expected_raw(fit, smap, 121, device="auto")
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(fit, k), v)
    assert smap.tolist() == [0, 0]


def test_public_package_exports_new_api_at_root_not_legacy():
    root = (SOURCE.parent / "__init__.py").read_text()
    legacy = (SOURCE.parent / "_legacy_init.py").read_text()
    assert "score_two_tier_grm_expected_raw as score_two_tier_grm_expected_raw" in root
    assert "TwoTierGrmExpectedRawScores as TwoTierGrmExpectedRawScores" in root
    assert '"score_two_tier_grm_expected_raw",' in root and '"TwoTierGrmExpectedRawScores",' in root
    # ADR-0028 rule 5: _legacy_init.py is deprecation-only; no new names there.
    assert "score_two_tier_grm_expected_raw" not in legacy
    assert "TwoTierGrmExpectedRawScores" not in legacy


def test_binding_registers_and_returns_native_flag():
    source = (SOURCE.parents[2] / "crates" / "fast-mlsirm-py" / "src" / "lib.rs").read_text()
    assert "wrap_pyfunction!(two_tier_expected_raw_with_provenance, m)" in source
    start = source.index("fn two_tier_expected_raw_with_provenance(")
    body = source[start:source.index("\n}\n", start)]
    assert "used_gpu" in body and "_used_gpu" not in body


def test_native_provenance_parity_when_binding_available():
    """Real compiled core: CPU path is bit-identical and never reports GPU;
    GPU/auto requests report a native bool and stay within the #2271 derived
    f32 bound when the GPU ran, or are bit-identical when they fell back.
    Uses fixed synthetic parameters (no fit) and q_specific=121."""
    import fast_mlsirm as fm

    assert hasattr(fm._core, "two_tier_expected_raw_with_provenance"), (
        "compiled core predates two_tier_expected_raw_with_provenance; rebuild it"
    )
    rng = np.random.default_rng(20261006)
    n_items, n_primary, n_specific, n_cat, q = 6, 2, 2, 3, 121
    a_primary = np.zeros((n_items, n_primary))
    a_primary[:, 0] = rng.uniform(0.8, 1.6, n_items)
    a_primary[2:4, 1] = rng.uniform(0.5, 1.0, 2)
    fit = SimpleNamespace(
        a_primary=a_primary, a_specific=rng.uniform(0.3, 0.8, n_items),
        threshold=np.sort(rng.uniform(-1.5, 1.5, (n_items, n_cat - 1)), axis=1)[:, ::-1].copy(),
        theta_p_eap=rng.normal(size=(64, n_primary)),
        n_primary=n_primary, n_specific=n_specific, n_cat=n_cat,
    )
    smap = np.array([0, 0, 1, 1, 0, 1])
    legacy_cpu = fm.expected_raw_two_tier_grm(fit, smap, q, device="cpu")
    cpu = fm.score_two_tier_grm_expected_raw(fit, smap, q, device="cpu")
    assert cpu.used_gpu is False
    np.testing.assert_array_equal(cpu.expected_raw, legacy_cpu)
    max_score = n_items * (n_cat - 1)
    bound = (8 + n_items * (n_cat - 1) * q) * 2.0**-24 * max_score
    require_gpu = os.environ.get("FAST_MLSIRM_REQUIRE_SCORE_GPU") is not None
    for device in ("auto", "gpu"):
        result = fm.score_two_tier_grm_expected_raw(fit, smap, q, device=device)
        assert type(result.used_gpu) is bool and result.device_requested == device
        if require_gpu:
            assert result.used_gpu, f"device={device}: hardware run requires actual GPU execution"
        if result.used_gpu:
            np.testing.assert_allclose(result.expected_raw, legacy_cpu, rtol=0.0, atol=bound,
                                       err_msg=f"device={device} used_gpu=True")
        else:
            np.testing.assert_array_equal(result.expected_raw, legacy_cpu,
                                          err_msg=f"device={device} used_gpu=False")


@pytest.mark.parametrize("field,value,match", [
    ("a_specific", np.ones((2, 1)), "fit.a_specific must be a 1-D"),
    ("a_primary", np.ones((2, 3)), "fit.a_primary must have shape"),
    ("threshold", np.ones((2, 1)), "fit.threshold must have shape"),
    ("theta_p_eap", np.ones((3, 3)), "fit.theta_p_eap must have shape"),
    ("theta_p_eap", np.array([[0.0, np.nan]]), "fit.theta_p_eap must be finite"),
])
@pytest.mark.parametrize("entry", ["score_two_tier_grm_expected_raw", "expected_raw_two_tier_grm"])
def test_shared_fit_validation_on_both_paths(monkeypatch, fit, entry, field, value, match):
    core, calls = _recording_core(False)
    module = _load(monkeypatch, core)
    bad = SimpleNamespace(**{**vars(fit), field: value})
    with pytest.raises(ValueError, match=match):
        getattr(module, entry)(bad, np.array([0, 0]), 121, device="cpu")
    assert calls == {"values": [], "provenance": []}


def test_legacy_missing_core_fails_closed(monkeypatch, fit):
    module = _load(monkeypatch, None)
    with pytest.raises(RuntimeError, match="compiled Rust core"):
        module.expected_raw_two_tier_grm(fit, np.array([0, 0]), 121, device="cpu")


def test_result_vector_is_read_only(monkeypatch, fit):
    core, _ = _recording_core(False)
    module = _load(monkeypatch, core)
    result = module.score_two_tier_grm_expected_raw(fit, np.array([0, 0]), 121, device="cpu")
    assert result.expected_raw.flags.writeable is False
    with pytest.raises(ValueError):
        result.expected_raw[0] = 0.0


def test_public_name_keeps_model_token_after_verb():
    """ADR-0028 rules 2 and 4: <verb>_<model token>_<object>; the model token
    two_tier_grm must not be split by the object."""
    root = (SOURCE.parent / "__init__.py").read_text()
    assert "score_two_tier_expected_raw_grm" not in root
    assert "score_two_tier_grm_expected_raw" in root
