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
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

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
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

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
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)

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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None)
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
    receipt = module.build_receipt(consumer_sha=None, build_source_sha=None, device="gpu")

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
    receipt = module.build_receipt(None, None, device="gpu")
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
    receipt = module.build_receipt(None, None, device="gpu")
    assert receipt["gpu_execution_used"] is True
    assert receipt["gpu_backend"] == "Metal"
    assert receipt["all_pass"] is True


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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
    assert receipt["anchor_rows_fixed"] is False
    assert receipt["all_pass"] is False


@pytest.mark.parametrize("field", ["consumer_sha", "build_source_sha"])
@pytest.mark.parametrize("invalid", ["abc", "A" * 40, "g" * 40, 42])
def test_invalid_revision_is_rejected_before_receipt_work(monkeypatch, field, invalid) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_sha256", lambda _: pytest.fail("invalid revision reached receipt work"))
    revisions = {"consumer_sha": None, "build_source_sha": None, field: invalid}
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        module.build_receipt(**revisions)


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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
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
    receipt = module.build_receipt(None, None)
    assert receipt["wrong_model_reject"] is True
    assert receipt["all_pass"] is True


def test_valid_revision_labels_and_null_handling_are_preserved(monkeypatch) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_core", _fake_core(module))
    receipt = module.build_receipt("a" * 40, "0" * 40)
    assert receipt["sha"] == "a" * 40
    assert receipt["build_source_sha"] == "0" * 40
    assert receipt["build_source_sha_present"] is True


def test_cli_refuses_unsanitized_nonfinite_json(monkeypatch, capsys) -> None:
    module = _load_script()
    monkeypatch.setattr(module, "_arguments", lambda: SimpleNamespace(
        consumer_sha=None, build_source_sha=None, device="cpu"))
    monkeypatch.setattr(module, "build_receipt", lambda *args: {"row_order_max_abs": np.nan})
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
    receipt = module.build_receipt(None, None)
    assert receipt["convergence_failure"] is expected
    if not expected:
        assert receipt["all_pass"] is False
