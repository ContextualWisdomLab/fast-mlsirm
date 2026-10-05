"""Receipt contract for the S1 two-tier GRM FIPC consumer verification script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (ROOT / "contracts" / "s1-fipc-consumer-verify-v1.schema.json").read_text()
)


def _load_script():
    script = ROOT / "scripts" / "s1_fipc_consumer_verify.py"
    spec = importlib.util.spec_from_file_location("s1_fipc_consumer_verify", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _assert_schema(receipt: dict) -> None:
    properties = SCHEMA["properties"]
    assert set(SCHEMA["required"]) <= receipt.keys()
    assert receipt.keys() <= properties.keys()
    assert receipt["row_order_diagnosis"] in properties["row_order_diagnosis"]["enum"]


def _fake_core(module):
    """A binding stand-in that returns flattened arrays, as the PyO3 layer does."""
    m = module

    def fit(*_args, **_kwargs):
        return {
            "converged": True,
            "a_primary": m.TRUE_A_P.reshape(-1).copy(),
            "a_specific": m.TRUE_A_S.copy(),
            "threshold": m.TRUE_D.reshape(-1).copy(),
        }

    def fipc(y, *args):
        fixed_a_p, fixed_a_s, fixed_d, max_iter = args[9], args[10], args[11], args[14]
        n_persons = y.size // m.N_ITEMS
        # Row-equivariant stand-in: each person's EAP depends only on their own row.
        theta = np.column_stack((y.reshape(n_persons, m.N_ITEMS)[:, :3].sum(axis=1),
                                 y.reshape(n_persons, m.N_ITEMS)[:, 3:].sum(axis=1))) * 0.1
        converged = max_iter > 1
        return {
            "converged": converged,
            "termination_reason": "tolerance" if converged else "max_iter_reached",
            "theta_p_eap": theta.reshape(-1),
            "a_primary": np.asarray(fixed_a_p).copy(),
            "a_specific": np.asarray(fixed_a_s).copy(),
            "threshold": np.asarray(fixed_d).copy(),
            "primary_mean": np.array([0.6, -0.3]),
            "primary_sd": np.array([1.2, 0.85]),
            "specific_sd": np.ones(m.N_SPECIFIC),
        }

    return SimpleNamespace(__file__=m._core.__file__, fit_two_tier_grm=fit,
                           fit_two_tier_grm_fipc=fipc)


def test_receipt_reports_missing_binding_honestly(monkeypatch) -> None:
    """Without the FIPC binding the receipt says so instead of hiding a TypeError."""
    module = _load_script()
    monkeypatch.delattr(module._core, "fit_two_tier_grm_fipc", raising=False)
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, q_primary=7, q_specific=7, q_expected_raw=31)

    _assert_schema(receipt)
    assert receipt["row_order_diagnosis"] == "binding_unavailable"
    assert receipt["anchor_identified"] is True
    assert receipt["row_order_max_abs"] is None
    assert receipt["build_source_sha_authority"] == "caller_asserted"
    assert receipt["all_pass"] is False
    assert "fipc_fit_error" not in receipt


def test_rank_deficient_anchor_fails_closed_before_fitting(monkeypatch) -> None:
    """Anchors loading on only one primary leave the other focal mean unidentified."""
    module = _load_script()
    monkeypatch.setattr(module, "ANCHOR", np.array([1, 1, 1, 0, 0, 0], dtype=bool))
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, q_primary=7, q_specific=7, q_expected_raw=31)

    _assert_schema(receipt)
    assert receipt["anchored_primary_rank"] == 1
    assert receipt["anchor_identified"] is False
    assert receipt["row_order_diagnosis"] == "not_identified"
    assert receipt["row_order_max_abs"] is None
    assert receipt["all_pass"] is False


def test_specific_variance_estimation_needs_an_anchor_per_specific_block() -> None:
    module = _load_script()
    anchor = np.array([1, 0, 0, 1, 0, 0], dtype=bool)  # spans both primaries, only specific 0
    assert module.anchor_identification(module.PRIMARY_MAP, module.SPECIFIC_MAP, anchor, False)["anchor_identified"]
    assert not module.anchor_identification(module.PRIMARY_MAP, module.SPECIFIC_MAP, anchor, True)["anchor_identified"]


def test_flattened_binding_outputs_are_reshaped_per_person(monkeypatch) -> None:
    """theta_p_eap arrives as n_persons*n_primary; the gates must work per person."""
    module = _load_script()
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, q_primary=7, q_specific=7, q_expected_raw=31)

    _assert_schema(receipt)
    assert "fipc_fit_error" not in receipt, receipt.get("fipc_fit_error")
    assert receipt["row_order_diagnosis"] == "pass"
    assert receipt["row_order_max_abs"] == 0.0
    assert receipt["anchor_rows_fixed"] is True
    assert receipt["responses_fit_eap_expected_raw"] is True
    assert receipt["focal_primary_mean_error"] == pytest.approx([0.6 - 0.65, -0.3 + 0.35])


@pytest.mark.parametrize("convergence", [False, None, 1, "true", "false", "missing"])
def test_reference_requires_convergence_before_focal_fit(monkeypatch, convergence) -> None:
    """An uncalibrated reference bank cannot start focal acceptance work."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm

    def reference(*args, **kwargs):
        result = original(*args, **kwargs)
        if convergence == "missing":
            result.pop("converged")
        else:
            result["converged"] = convergence
        return result

    fake.fit_two_tier_grm = reference
    fake.fit_two_tier_grm_fipc = lambda *args, **kwargs: pytest.fail("invalid reference reached focal fitting")
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


def test_nonconverged_permutation_cannot_certify_row_order(monkeypatch) -> None:
    """Equal EAP rows do not certify a refit that exhausted its step budget."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == 2:
            fit.update(converged=False, termination_reason="step_limited")
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["row_order_max_abs"] == 0.0
    assert receipt["row_order"] is False
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("call_number", [1, 2])
def test_nonfinite_specific_loading_cannot_pass_receipt(monkeypatch, invalid, call_number) -> None:
    """A free specific loading must be finite in either row-order fit."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit["a_specific"][2] = invalid
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("convergence", [1, "true", "false", None])
def test_truthy_convergence_cannot_certify_responses(monkeypatch, convergence) -> None:
    """Only an explicit binding boolean certifies converged response scoring."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args):
        fit = original(y, *args)
        if fit["converged"]:
            fit["converged"] = convergence
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["responses_fit_eap_expected_raw"] is False
    assert receipt["row_order"] is False
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("field", ["theta_p_eap", "a_primary", "a_specific", "threshold"])
@pytest.mark.parametrize("call_number", [1, 2])
def test_nonfinite_fit_output_cannot_pass_receipt(monkeypatch, field, call_number) -> None:
    """Every item/EAP output is finite-checked in both row-order fits."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit[field][0] = np.nan
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("call_number", [1, 2])
def test_missing_permutation_convergence_fails_with_finite_diagnostics(monkeypatch, call_number) -> None:
    """Missing convergence cannot pass, while computed finite row drift remains visible."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit.pop("converged")
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["row_order"] is False
    assert receipt["row_order_max_abs"] == 0.0
    assert receipt["row_order_rmse"] == 0.0
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("field", ["primary_mean", "primary_sd"])
@pytest.mark.parametrize("invalid", [0.6, [0.6], [0.6, 0.7, 0.8], [[0.6, 0.7]], [np.nan, 0.7]])
@pytest.mark.parametrize("call_number", [1, 2])
def test_primary_moment_shape_and_finiteness_fail_closed(monkeypatch, field, invalid, call_number) -> None:
    """Neither refit may broadcast malformed focal moments into recovery errors."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit[field] = invalid
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("invalid", [[0.0, 0.8], [-1.2, 0.8], [np.inf, 0.8]])
@pytest.mark.parametrize("call_number", [1, 2])
def test_nonpositive_or_nonfinite_primary_sd_fails_closed(monkeypatch, invalid, call_number) -> None:
    """A focal SD must be positive and finite in both refits."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit["primary_sd"] = invalid
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


def test_non_unit_focal_prior_detects_shrinking_sd(monkeypatch) -> None:
    """A focal SD below 1 moves the prior off the unit reference too."""
    module = _load_script()
    fake = _fake_core(module)
    base_fipc = fake.fit_two_tier_grm_fipc

    def fipc(*args):
        fit = base_fipc(*args)
        fit["primary_sd"] = np.array([1.0, 0.8])
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["non_unit_focal_prior"] is True


def test_gpu_request_fails_closed_on_cpu_fallback(monkeypatch) -> None:
    """A GPU receipt only passes when the binding reports GPU execution."""
    module = _load_script()
    fake = _fake_core(module)
    base_fipc = fake.fit_two_tier_grm_fipc
    seen: list[str] = []

    def fipc(*args, device="cpu"):
        seen.append(device)
        fit = base_fipc(*args)
        fit.update(gpu_execution_used=False, gpu_backend=None, gpu_device_name=None)
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, device="gpu", q_primary=7, q_specific=7, q_expected_raw=31)

    _assert_schema(receipt)
    assert set(seen) == {"gpu"}
    assert receipt["device_requested"] == "gpu"
    assert receipt["gpu_execution_used"] is False
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("call_number", [1, 2, 3])
@pytest.mark.parametrize("gpu_used", [False, None])
def test_gpu_receipt_requires_dispatch_in_every_valid_fit(monkeypatch, call_number, gpu_used) -> None:
    """A single CPU fallback or missing dispatch cannot certify the GPU receipt."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args, device="cpu"):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        fit.update(gpu_execution_used=True, gpu_backend="Metal")
        if calls == call_number:
            fit["gpu_execution_used"] = gpu_used
            fit["gpu_backend"] = None
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device="gpu", q_primary=7, q_specific=7, q_expected_raw=31)
    assert calls == 3
    assert receipt["gpu_execution_used"] is False
    assert receipt["gpu_backend"] is None
    assert receipt["all_pass"] is False


def test_gpu_receipt_accepts_dispatch_in_every_valid_fit(monkeypatch) -> None:
    """The diagnostic still passes when all valid fits explicitly dispatch."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args, device="cpu"):
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        fit = original(y, *args)
        fit.update(gpu_execution_used=True, gpu_backend="Metal")
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device="gpu", q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["gpu_execution_used"] is True
    assert receipt["gpu_backend"] == "Metal"
    assert receipt["all_pass"] is True


@pytest.mark.parametrize("fallback_call", [None, 1, 2, 3])
def test_auto_reports_all_call_dispatch_without_rejecting_fallback(monkeypatch, fallback_call) -> None:
    """Auto permits CPU fallback but must not label mixed execution as all-GPU."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args, device="cpu"):
        nonlocal calls
        assert device == "auto"
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        used = calls != fallback_call
        fit.update(gpu_execution_used=used, gpu_backend="Metal" if used else None)
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device="auto", q_primary=7, q_specific=7, q_expected_raw=31)
    assert calls == 3
    assert receipt["gpu_execution_used"] is (fallback_call is None)
    assert receipt["gpu_backend"] == ("Metal" if fallback_call is None else None)
    assert receipt["all_pass"] is True


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_nonstring_gpu_backend_cannot_pass_schema_receipt(monkeypatch, device) -> None:
    """A binding metadata object cannot become a schema-invalid PASS receipt."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args, **kwargs):
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        fit = original(y, *args)
        fit.update(gpu_execution_used=True, gpu_backend={"unexpected": "object"})
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    assert receipt["gpu_backend"] is None
    assert receipt["gpu_execution_used"] is None
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("call_number", [1, 2, 3])
@pytest.mark.parametrize("field, invalid", [
    ("gpu_execution_used", 1), ("gpu_backend", {"unexpected": "object"}),
])
def test_dispatch_metadata_schema_checks_every_valid_call(monkeypatch, device, call_number, field, invalid) -> None:
    """Aggregation must not hide malformed metadata from either later call."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args, **kwargs):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        fit.update(gpu_execution_used=True, gpu_backend="Metal")
        if calls == call_number:
            fit[field] = invalid
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["gpu_execution_used"] is None
    assert receipt["gpu_backend"] is None


def test_consumer_revision_comes_from_script_checkout(monkeypatch, tmp_path) -> None:
    module = _load_script()
    expected = module.subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    module.subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    module.subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=Fixture",
         "-c", "user.email=fixture@example.invalid", "commit", "-q",
         "--allow-empty", "-m", "Unrelated checkout"], check=True,
    )
    monkeypatch.chdir(tmp_path)
    assert module._git_sha() == expected


@pytest.mark.parametrize("field", ["a_primary", "a_specific", "threshold"])
@pytest.mark.parametrize("call_number", [1, 2])
def test_anchor_gate_checks_both_refits(monkeypatch, field, call_number) -> None:
    """Changing an anchor in either refit violates the fixed-bank contract."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit[field][0] += 0.1
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["anchor_rows_fixed"] is False
    assert receipt["all_pass"] is False
    assert receipt["row_order_max_abs"] == 0.0


@pytest.mark.parametrize("call_number", [1, 2])
def test_specific_prior_gate_checks_both_refits(monkeypatch, call_number) -> None:
    """A changed fixed specific prior in either refit cannot pass."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = 0

    def fipc(y, *args):
        nonlocal calls
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls += 1
        fit = original(y, *args)
        if calls == call_number:
            fit["specific_sd"][0] = 2.0
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["orthogonal_specific_prior_fixed"] is False
    assert receipt["all_pass"] is False
    assert receipt["row_order_max_abs"] == 0.0


def test_anchor_gate_detects_changed_specific_loading(monkeypatch) -> None:
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(*args):
        fit = original(*args)
        fit["a_specific"][0] += 0.1
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["anchor_rows_fixed"] is False
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("field", ["consumer_sha", "build_source_sha"])
@pytest.mark.parametrize("invalid", ["abc", "A" * 40, "g" * 40, 42])
def test_invalid_revision_is_rejected_before_receipt_work(monkeypatch, field, invalid) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_sha256", lambda _: pytest.fail("invalid revision reached receipt work"))
    revisions = {"consumer_sha": None, "build_source_sha": None, field: invalid}
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        module.build_receipt(**revisions, q_primary=7, q_specific=7, q_expected_raw=31)


@pytest.mark.parametrize("field", ["theta_p_eap", "primary_mean", "primary_sd", "a_primary"])
def test_nonfinite_fit_metrics_remain_failed_standard_json(monkeypatch, field) -> None:
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(*args):
        fit = original(*args)
        fit[field][4 if field == "a_primary" else 0] = np.nan
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    json.dumps(receipt, allow_nan=False)
    _assert_schema(receipt)
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("error_type", [RuntimeError, OSError, ValueError])
def test_wrong_map_gate_does_not_certify_runtime_errors(monkeypatch, error_type) -> None:
    """Internal failures mentioning a primary/map are not validation evidence."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args):
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise error_type("primary map kernel runtime failure")
        return original(y, *args)

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["wrong_model_reject"] is False
    assert receipt["all_pass"] is False


def test_wrong_map_gate_certifies_documented_validation(monkeypatch) -> None:
    """The observed PyO3 empty-dimension ValueError remains accepted."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args):
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        return original(y, *args)

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["wrong_model_reject"] is True
    assert receipt["all_pass"] is True


@pytest.mark.parametrize("device", ["invalid_device", "CPU", "", None, True, 1, []])
def test_invalid_device_is_rejected_before_receipt_work(monkeypatch, device) -> None:
    """Direct callers must obey the same device enum as the CLI and schema."""
    module = _load_script()
    monkeypatch.setattr(module, "_sha256", lambda _: pytest.fail("invalid device reached receipt work"))
    with pytest.raises(ValueError, match="Device must be one of cpu, gpu, auto"):
        module.build_receipt(None, None, device=device, q_primary=7, q_specific=7, q_expected_raw=31)


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_valid_device_enum_preserves_receipt_acceptance(monkeypatch, device) -> None:
    """All documented devices remain accepted; dispatch is a binding stand-in."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        fit = original(y, *args)
        fit.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=7, q_specific=7, q_expected_raw=31)
    _assert_schema(receipt)
    assert receipt["device_requested"] == device
    assert receipt["all_pass"] is True


def test_valid_revision_labels_and_null_handling_are_preserved(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt("a" * 40, "0" * 40, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["sha"] == "a" * 40
    assert receipt["build_source_sha"] == "0" * 40
    assert receipt["build_source_sha_present"] is True


def test_cli_refuses_unsanitized_nonfinite_json(monkeypatch, capsys) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_arguments", lambda: SimpleNamespace(
        consumer_sha=None, build_source_sha=None, device="cpu",
        q_primary=7, q_specific=7, q_expected_raw=31))
    monkeypatch.setattr(module, "build_receipt", lambda *args, **kwargs: {"row_order_max_abs": np.nan})
    with pytest.raises(ValueError, match="JSON compliant"):
        module.main()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "reason, reported_convergence, expected",
    [("max_iter_reached", False, True), ("step_limited", False, True),
     ("unknown_stop", False, False), (None, False, False),
     ("max_iter_reached", None, False), ("max_iter_reached", 0, False),
     ("step_limited", True, False)],
)
def test_budget_failure_requires_documented_nonconvergence(monkeypatch, reason, reported_convergence, expected) -> None:
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc

    def fipc(*args):
        fit = original(*args)
        if not fit["converged"]:
            if reason is None:
                fit.pop("termination_reason")
            else:
                fit["termination_reason"] = reason
            fit["converged"] = reported_convergence
        return fit

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=7, q_specific=7, q_expected_raw=31)
    assert receipt["convergence_failure"] is expected
    if not expected:
        assert receipt["all_pass"] is False


BUDGET_OUTPUT_FIELDS = (
    "theta_p_eap", "a_primary", "a_specific", "threshold", "primary_mean", "primary_sd",
)


def _budget_output_receipt(monkeypatch, device, field=None, missing=False):
    """Mutate only the third valid call, keeping both row-order fits valid."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError(
                "primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required"
            )
        result = original(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(
            gpu_execution_used=device != "cpu",
            gpu_backend="Metal" if device != "cpu" else None,
        )
        if len(calls) == 3 and field is not None:
            assert result["converged"] is False
            assert result["termination_reason"] == "max_iter_reached"
            if missing:
                result.pop(field)
            else:
                result[field][0] = np.nan
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(
        None, None, device=device, q_primary=121, q_specific=121, q_expected_raw=121,
    )
    _assert_schema(receipt)
    json.dumps(receipt, allow_nan=False)
    assert calls == [(100, (121, 121)), (100, (121, 121)), (1, (121, 121))]
    return receipt


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("field", BUDGET_OUTPUT_FIELDS)
def test_nonfinite_budget_return_cannot_certify_negative_control(monkeypatch, device, field):
    receipt = _budget_output_receipt(monkeypatch, device, field=field)
    assert receipt["all_pass"] is False, "nonfinite third-call arrays certified all_pass"
    assert receipt["convergence_failure"] is False
    assert receipt["row_order_diagnosis"] == "fit_error"


@pytest.mark.parametrize("field", BUDGET_OUTPUT_FIELDS)
def test_missing_budget_field_cannot_certify_negative_control(monkeypatch, field):
    receipt = _budget_output_receipt(monkeypatch, "cpu", field=field, missing=True)
    assert receipt["all_pass"] is False, "incomplete third-call arrays certified all_pass"
    assert receipt["convergence_failure"] is False
    assert receipt["row_order_diagnosis"] == "fit_error"


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_finite_documented_budget_negative_remains_valid(monkeypatch, device):
    receipt = _budget_output_receipt(monkeypatch, device)
    assert receipt["all_pass"] is True
    assert receipt["convergence_failure"] is True
    assert receipt["row_order_diagnosis"] == "pass"


def _third_call_receipt(monkeypatch, device, field=None):
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError(
                "primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required"
            )
        result = original(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        if len(calls) == 3 and field is not None:
            assert result["converged"] is False
            assert result["termination_reason"] == "max_iter_reached"
            # Every value stays finite. Base/permutation remain unchanged and
            # equal; mutate only an anchored row or the fixed specific SD.
            result[field][0] += 0.125
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(
        None, None, device=device, q_primary=121, q_specific=121, q_expected_raw=121,
    )
    json.dumps(receipt, allow_nan=False)
    _assert_schema(receipt)
    assert calls == [(100, (121, 121)), (100, (121, 121)), (1, (121, 121))]
    assert receipt["convergence_failure"] is True
    assert receipt["row_order"] is True
    assert receipt["row_order_diagnosis"] == "pass"
    assert receipt["row_order_max_abs"] == 0.0
    return receipt


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("field", ["a_primary", "a_specific", "threshold", "specific_sd"])
def test_budget_return_must_preserve_fixed_contract(monkeypatch, device, field):
    receipt = _third_call_receipt(monkeypatch, device, field)
    assert receipt["all_pass"] is False, "changed third-call fixed bank/prior certified all_pass"
    gate = "orthogonal_specific_prior_fixed" if field == "specific_sd" else "anchor_rows_fixed"
    assert receipt[gate] is False


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_valid_budget_preserves_fixed_bank_and_prior(monkeypatch, device):
    receipt = _third_call_receipt(monkeypatch, device)
    assert receipt["anchor_rows_fixed"] is True
    assert receipt["orthogonal_specific_prior_fixed"] is True
    assert receipt["all_pass"] is True


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("call_number", [1, 2, 3])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_present_fixed_specific_trace_must_be_finite(monkeypatch, device, call_number, invalid):
    """Synthetic trace corruption follows the actual Q481 diagnostic field."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = original(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        result["fixed_specific_second_moment_trace"] = [9.875, 9.875]
        if len(calls) == call_number:
            result["fixed_specific_second_moment_trace"][1] = invalid
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=481, q_specific=241, q_expected_raw=121)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False, "nonfinite present native trace certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert "fixed_specific_second_moment_trace" in receipt["fipc_fit_error"]
    assert all(q == (481, 241) for _, q in calls)


OPTIONAL_NATIVE_DIAGNOSTICS = (
    "theta_p_sd", "primary_cov", "loglik_trace", "fixed_loglik_trace",
    "fixed_primary_first_moment_trace", "fixed_primary_second_moment_trace",
    "prior_mean_trace", "prior_covariance_trace", "prior_specific_sd_trace",
    "final_loglik_change", "final_param_change",
)


@pytest.mark.parametrize("field", OPTIONAL_NATIVE_DIAGNOSTICS)
@pytest.mark.parametrize("call_number", [1, 2, 3])
def test_present_sibling_native_diagnostic_must_be_finite(monkeypatch, field, call_number):
    """Present exported numeric diagnostics cannot evade shared call checks."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args):
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = original(y, *args)
        calls.append(args[12:14])
        result[field] = np.nan if field.startswith("final_") else [0.5, np.nan]
        if len(calls) != call_number:
            result[field] = 0.5 if field.startswith("final_") else [0.5, 0.5]
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=121, q_specific=241, q_expected_raw=121)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False, "nonfinite sibling diagnostic certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert field in receipt["fipc_fit_error"]
    assert all(pair == (121, 241) for pair in calls)


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
def test_present_finite_native_diagnostics_preserve_acceptance(monkeypatch, device):
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = original(y, *args)
        calls.append(args[12:14])
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        for field in OPTIONAL_NATIVE_DIAGNOSTICS + ("fixed_specific_second_moment_trace",):
            result[field] = 0.5 if field.startswith("final_") else [0.5, 0.5]
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=121, q_specific=241, q_expected_raw=121)
    assert receipt["all_pass"] is True
    assert receipt["row_order_diagnosis"] == "pass"
    assert calls == [(121, 241)] * 3
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_reference_present_specific_trace_rejected_before_focal(monkeypatch, invalid):
    """A converged label cannot conceal nonfinite reference diagnostics."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm
    focal_calls = []
    focal_original = fake.fit_two_tier_grm_fipc

    def reference(*args, **kwargs):
        result = original(*args, **kwargs)
        result["fixed_specific_second_moment_trace"] = [0.5, invalid]
        return result

    def focal(y, *args, **kwargs):
        focal_calls.append(args[12:14])
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        return focal_original(y, *args)

    fake.fit_two_tier_grm = reference
    fake.fit_two_tier_grm_fipc = focal
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=121, q_specific=241, q_expected_raw=121)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False, "nonfinite reference trace certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert "fixed_specific_second_moment_trace" in receipt["fipc_fit_error"]
    assert focal_calls == []


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("diagnostics_present", [False, True])
def test_finite_or_older_reference_diagnostics_preserve_acceptance(monkeypatch, device, diagnostics_present):
    module = _load_script()
    fake = _fake_core(module)
    ref_original = fake.fit_two_tier_grm
    focal_original = fake.fit_two_tier_grm_fipc
    focal_calls = []

    def reference(*args, **kwargs):
        result = ref_original(*args, **kwargs)
        if diagnostics_present:
            for field in OPTIONAL_NATIVE_DIAGNOSTICS + ("fixed_specific_second_moment_trace",):
                result[field] = 0.5 if field.startswith("final_") else [0.5, 0.5]
        return result

    def focal(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = focal_original(y, *args)
        focal_calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        return result

    fake.fit_two_tier_grm = reference
    fake.fit_two_tier_grm_fipc = focal
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=121, q_specific=241, q_expected_raw=121)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is True
    assert focal_calls == [(100, (121, 241)), (100, (121, 241)), (1, (121, 241))]


@pytest.mark.parametrize("field", OPTIONAL_NATIVE_DIAGNOSTICS)
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_reference_present_sibling_diagnostic_rejected_before_focal(monkeypatch, field, invalid):
    """Every present reference diagnostic shares the focal finite boundary."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm
    focal_calls = []
    focal_original = fake.fit_two_tier_grm_fipc

    def reference(*args, **kwargs):
        result = original(*args, **kwargs)
        result[field] = invalid if field.startswith("final_") else [0.5, invalid]
        return result

    def focal(y, *args, **kwargs):
        focal_calls.append(args[12:14])
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        return focal_original(y, *args)

    fake.fit_two_tier_grm = reference
    fake.fit_two_tier_grm_fipc = focal
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, q_primary=121, q_specific=241, q_expected_raw=121)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False, "nonfinite reference sibling certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert field in receipt["fipc_fit_error"]
    assert focal_calls == []


STOP_CONTRADICTION_REASONS = ("step_limited", "max_iter_reached", "prior_update_stalled")


def _termination_metadata_receipt(monkeypatch, device, position, reason):
    """Record synthetic contradictory labels without invoking native fitting."""
    module = _load_script()
    fake = _fake_core(module)
    reference = fake.fit_two_tier_grm
    focal = fake.fit_two_tier_grm_fipc
    calls = []

    def mutate(result):
        assert result["converged"] is True
        if reason is None:
            result.pop("termination_reason", None)
        else:
            result["termination_reason"] = reason
        return result

    def reference_return(*args, **kwargs):
        result = reference(*args, **kwargs)
        return mutate(result) if position == "reference" else result

    def focal_return(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError(
                "primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required"
            )
        result = focal(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        target = 1 if position == "base" else 2
        if position != "reference" and len(calls) == target:
            mutate(result)
        return result

    fake.fit_two_tier_grm = reference_return
    fake.fit_two_tier_grm_fipc = focal_return
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(
        None, None, device=device, q_primary=121, q_specific=121, q_expected_raw=121,
    )
    _assert_schema(receipt)
    json.dumps(receipt, allow_nan=False)
    return receipt, calls


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("position", ["reference", "base", "permuted"])
@pytest.mark.parametrize("reason", STOP_CONTRADICTION_REASONS)
def test_converged_label_cannot_override_present_nonconvergence_reason(monkeypatch, device, position, reason):
    """An explicit failed-stop label cannot certify an otherwise valid receipt."""
    receipt, calls = _termination_metadata_receipt(monkeypatch, device, position, reason)
    assert receipt["all_pass"] is False, "contradictory native stop labels certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["fipc_fit_error"] == "ValueError: Native convergence metadata contradicts termination reason."
    count = {"reference": 0, "base": 1, "permuted": 2}[position]
    assert calls == [(100, (121, 121))] * count
    assert receipt["gpu_execution_used"] is None
    assert receipt["gpu_backend"] is None


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("position", ["reference", "base", "permuted"])
@pytest.mark.parametrize("reason", [None, "tolerance", "tolerance_met"])
def test_compatible_converged_stop_metadata_preserves_receipt(monkeypatch, device, position, reason):
    """Absent older exports and both existing tolerance spellings remain valid."""
    receipt, calls = _termination_metadata_receipt(monkeypatch, device, position, reason)
    assert receipt["all_pass"] is True
    assert receipt["row_order_diagnosis"] == "pass"
    assert calls == [(100, (121, 121)), (100, (121, 121)), (1, (121, 121))]
    assert receipt["convergence_failure"] is True
    assert "fipc_fit_error" not in receipt


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("bank_case, expected_rank", [("zero", 0), ("first_axis_only", 1), ("second_axis_only", 1), ("signed_full", 2), ("scaled_full", 2), ("true_full", 2)])
def test_actual_fixed_anchor_rank_controls_focal_admission(monkeypatch, device, bank_case, expected_rank):
    """Actual anchor slopes, not their binary pattern, must span focal location.

    Kim (2006, pp. 359-360) bases fixed calibration on the old item scale.
    The rank necessity is derived here from the implemented A_anchor @ mu
    predictor; it is not a multidimensional theorem attributed to Kim.

    References
    ----------
    Kim, S. (2006). A comparative study of IRT fixed parameter calibration
        methods. Journal of Educational Measurement, 43(4), 355-381.
        https://doi.org/10.1111/j.1745-3984.2006.00021.x
    """
    module = _load_script()
    bank = module.TRUE_A_P.copy()
    if bank_case == "zero":
        bank[module.ANCHOR] = 0.0
    elif bank_case == "first_axis_only":
        bank[module.ANCHOR, 1] = 0.0
    elif bank_case == "second_axis_only":
        bank[module.ANCHOR, 0] = 0.0
    elif bank_case == "signed_full":
        bank[:, 0] *= -1.0
    elif bank_case == "scaled_full":
        bank *= np.array([0.25, 4.0])
    # Free slopes remain full-rank even in rank-deficient anchored banks.
    assert np.linalg.matrix_rank(bank[~module.ANCHOR]) == 2
    assert np.linalg.matrix_rank(bank[module.ANCHOR]) == expected_rank
    assert module.anchor_identification(module.PRIMARY_MAP, module.SPECIFIC_MAP, module.ANCHOR, False)["anchored_primary_rank"] == 2
    before = bank.copy()
    fake = _fake_core(module)
    reference = fake.fit_two_tier_grm
    focal = fake.fit_two_tier_grm_fipc
    calls = []

    def reference_return(*args, **kwargs):
        result = reference(*args, **kwargs)
        result["a_primary"] = bank.reshape(-1).copy()
        return result

    def focal_return(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        calls.append((args[14], args[12:14]))
        result = focal(y, *args)
        result.update(gpu_execution_used=device != "cpu", gpu_backend="Metal" if device != "cpu" else None)
        return result

    fake.fit_two_tier_grm = reference_return
    fake.fit_two_tier_grm_fipc = focal_return
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=121, q_specific=121, q_expected_raw=121)
    _assert_schema(receipt)
    json.dumps(receipt, allow_nan=False)
    if expected_rank < 2:
        assert receipt["all_pass"] is False, "rank-deficient actual fixed anchor bank certified all_pass"
        assert calls == [], "unidentified actual bank reached focal fitter"
        assert receipt["anchor_identified"] is False
        assert receipt["anchored_primary_rank"] == expected_rank
        assert receipt["row_order_diagnosis"] == "not_identified"
        assert receipt["row_order_max_abs"] is None
        assert receipt["gpu_execution_used"] is None
        assert all(receipt[key] is False for key in module.GATES)
    else:
        assert receipt["all_pass"] is True, receipt.get("fipc_fit_error")
        assert receipt["anchor_identified"] is True
        assert receipt["anchored_primary_rank"] == 2
        assert receipt["row_order_diagnosis"] == "pass"
        assert calls == [(100, (121, 121)), (100, (121, 121)), (1, (121, 121))]
        assert receipt["anchor_rows_fixed"] is True
        assert receipt["convergence_failure"] is True
    assert np.array_equal(bank, before)


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("call_number", [1, 2, 3])
@pytest.mark.parametrize("backend_case", ["missing", "null", "empty", "whitespace"])
def test_true_gpu_dispatch_requires_named_backend_each_call(monkeypatch, device, call_number, backend_case):
    """A true dispatch flag alone cannot certify an unidentified GPU backend."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = original(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=True, gpu_backend="Metal")
        if len(calls) == call_number:
            if backend_case == "missing":
                result.pop("gpu_backend")
            else:
                result["gpu_backend"] = {"null": None, "empty": "", "whitespace": " \t\n"}[backend_case]
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=121, q_specific=241, q_expected_raw=121)
    _assert_schema(receipt)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is False, "true GPU flag with unidentified backend certified all_pass"
    assert receipt["row_order_diagnosis"] == "fit_error"
    assert receipt["fipc_fit_error"] == "ValueError: GPU execution requires a nonempty backend name."
    assert receipt["gpu_execution_used"] is None
    assert receipt["gpu_backend"] is None
    assert calls == [(100 if i < 2 else 1, (121, 241)) for i in range(call_number)]


@pytest.mark.parametrize("device", ["cpu", "gpu", "auto"])
@pytest.mark.parametrize("backend", ["Metal", "Vulkan"])
def test_true_gpu_dispatch_named_backend_preserves_original_receipt(monkeypatch, device, backend):
    """Operational metadata validation does not restrict legitimate backend names."""
    module = _load_script()
    fake = _fake_core(module)
    original = fake.fit_two_tier_grm_fipc
    calls = []

    def fipc(y, *args, **kwargs):
        assert kwargs.get("device", "cpu") == device
        if not np.array_equal(args[1], module.PRIMARY_MAP.reshape(-1)):
            raise ValueError("primary dimension 1 has 0 loading item(s); at least two loading items per primary dimension are required")
        result = original(y, *args)
        calls.append((args[14], args[12:14]))
        result.update(gpu_execution_used=True, gpu_backend=backend)
        return result

    fake.fit_two_tier_grm_fipc = fipc
    monkeypatch.setattr(module, "_core", fake)
    receipt = module.build_receipt(None, None, device=device, q_primary=121, q_specific=241, q_expected_raw=121)
    _assert_schema(receipt)
    json.dumps(receipt, allow_nan=False)
    assert receipt["all_pass"] is True
    assert receipt["gpu_execution_used"] is True
    assert receipt["gpu_backend"] == backend
    assert calls == [(100, (121, 241)), (100, (121, 241)), (1, (121, 241))]


def test_expected_raw_q481_reference_is_finite_and_symmetric():
    """A symmetric four-category bank has total expectation nine at zero."""
    module = _load_script()
    theta = np.zeros((2, 2), dtype=np.float64)
    fit = {
        "a_primary": module.TRUE_A_P.copy(),
        "a_specific": np.array([.6, .7, -.8, .5, -.4, .9]),
        "threshold": np.tile(np.array([1.2, 0.0, -1.2]), (6, 1)),
    }
    actual = module.expected_raw(theta, fit, q_specific=481)
    assert actual.shape == (2,) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, [9.0, 9.0], atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("q", [1, 121, 241, 481, 961])
def test_normal_reference_rule_preserves_requested_count_and_moments(q):
    """Analytic moments check the standard-normal metric, not fit accuracy."""
    module = _load_script()
    nodes, weights = module._normal_reference_rule(np.int64(q))
    assert nodes.shape == weights.shape == (q,)
    assert nodes.dtype == weights.dtype == np.dtype("float64")
    assert np.isfinite(nodes).all() and np.isfinite(weights).all()
    assert (weights >= 0).all() and (np.diff(nodes) > 0).all()
    np.testing.assert_array_equal(nodes, -nodes[::-1])
    np.testing.assert_array_equal(weights, weights[::-1])
    np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-12, rtol=1e-12)
    if q == 1:
        np.testing.assert_array_equal(nodes, [0.0])
        np.testing.assert_array_equal(weights, [1.0])
    else:
        for degree, moment in [(2, 1.0), (4, 3.0), (6, 15.0), (8, 105.0)]:
            np.testing.assert_allclose(weights @ nodes**degree, moment, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("q", [0, -1, True, np.bool_(True), 1.0, "121", None])
def test_normal_reference_rule_rejects_invalid_count_before_solver(monkeypatch, q):
    module = _load_script()
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid count reached the eigensolver")
    monkeypatch.setattr(module.np.linalg, "eigh", forbidden)
    with pytest.raises(ValueError, match="positive integer"):
        module._normal_reference_rule(q)


@pytest.mark.parametrize("corruption", ["nodes_nan", "vectors_nan", "zero_mass"])
def test_normal_reference_rule_rejects_invalid_solver_output(monkeypatch, corruption):
    module = _load_script()
    def invalid(matrix):
        q = matrix.shape[0]
        nodes = np.arange(q, dtype=np.float64)
        vectors = np.eye(q)
        if corruption == "nodes_nan":
            nodes[0] = np.nan
        elif corruption == "vectors_nan":
            vectors[0, 0] = np.nan
        else:
            vectors[0] = 0.0
        return nodes, vectors
    monkeypatch.setattr(module.np.linalg, "eigh", invalid)
    with pytest.raises(ValueError, match="finite nodes and weights with positive mass"):
        module._normal_reference_rule(121)


@pytest.mark.parametrize("q", [121, 241])
def test_normal_reference_rule_matches_original_low_order_expected_raw(q):
    """A distinct category-probability calculation uses the prior NumPy rule."""
    module = _load_script()
    theta = np.array([[-1.4, .3], [.6, -.8], [1.2, 1.1]])
    fit = {"a_primary": module.TRUE_A_P.copy(),
           "a_specific": np.array([.6, .7, -.8, .5, -.4, .9]),
           "threshold": module.TRUE_D.copy()}
    nodes, weights = np.polynomial.hermite.hermgauss(q)
    nodes *= np.sqrt(2.0)
    weights /= np.sqrt(np.pi)
    reference = np.zeros(theta.shape[0])
    for item in range(6):
        base = theta @ fit["a_primary"][item]
        cum = 1.0 / (1.0 + np.exp(-(base[:, None, None] + fit["a_specific"][item] * nodes[None, :, None] + fit["threshold"][item])))
        probabilities = np.concatenate((1.0 - cum[..., :1], -np.diff(cum, axis=-1), cum[..., -1:]), axis=-1)
        reference += (probabilities @ np.arange(4)) @ weights
    actual = module.expected_raw(theta, fit, q_specific=q)
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


def test_expected_raw_general_only_bank_retains_analytic_curve():
    module = _load_script()
    theta = np.array([[-1.4, .3], [.6, -.8], [1.2, 1.1]])
    fit = {"a_primary": module.TRUE_A_P.copy(),
           "a_specific": np.zeros(6), "threshold": module.TRUE_D.copy()}
    reference = np.zeros(3)
    for item in range(6):
        eta = (theta @ fit["a_primary"][item])[:, None] + fit["threshold"][item]
        reference += (1.0 / (1.0 + np.exp(-eta))).sum(axis=1)
    actual = module.expected_raw(theta, fit, q_specific=481)
    np.testing.assert_array_equal(actual, reference)
