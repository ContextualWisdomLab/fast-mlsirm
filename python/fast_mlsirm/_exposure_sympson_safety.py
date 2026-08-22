"""Preserve the Rust Sympson-Hetter tolerance contract at the Python boundary.

The package historically narrowed the public tolerance domain to ``tol > 0``
even though the Rust implementation admits every finite ``tol >= 0``. This
adapter restores that exact semantic boundary without moving simulation,
exposure-rate updates, convergence arithmetic, or any other numerical work out
of Rust.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_USIZE_MAX = int(np.iinfo(np.uintp).max)


def install(exposure_module: ModuleType) -> None:
    """Wrap ``sympson_hetter`` so zero tolerance reaches Rust unchanged."""

    original: Callable[..., Any] = exposure_module.sympson_hetter

    @wraps(original)
    def safe_sympson_hetter(
        a: object,
        b: object,
        c: object | None = None,
        *,
        r_max: float = 0.25,
        test_length: int = 20,
        n_simulees: int = 1000,
        max_iter: int = 20,
        tol: float = 0.02,
        seed: int = 20250724,
        q_theta: int = 41,
    ) -> Any:
        """Validate the native non-negative tolerance domain before data work."""

        tol_value = exposure_module._as_real_scalar("tol", tol)
        if not np.isfinite(tol_value) or tol_value < 0.0:
            raise ValueError("tol must be finite and non-negative")
        if tol_value > 0.0:
            return original(
                a,
                b,
                c,
                r_max=r_max,
                test_length=test_length,
                n_simulees=n_simulees,
                max_iter=max_iter,
                tol=tol_value,
                seed=seed,
                q_theta=q_theta,
            )

        # The historical wrapper rejects exactly zero, so preserve the native
        # domain by performing the same validation/marshalling and dispatching
        # the unchanged normalized arguments directly to the Rust binding.
        test_length_value = exposure_module._as_int(
            "test_length", test_length, maximum=_USIZE_MAX
        )
        n_simulees_value = exposure_module._as_int(
            "n_simulees", n_simulees, maximum=_USIZE_MAX
        )
        max_iter_value = exposure_module._as_int(
            "max_iter", max_iter, maximum=_USIZE_MAX
        )
        seed_value = exposure_module._as_int("seed", seed, maximum=2**64 - 1)
        q_theta_value = exposure_module._as_int(
            "q_theta", q_theta, maximum=_USIZE_MAX
        )
        r_max_value = exposure_module._as_real_scalar("r_max", r_max)
        if not np.isfinite(r_max_value) or not (0.0 < r_max_value <= 1.0):
            raise ValueError("r_max must be finite and in (0, 1]")

        a_value = exposure_module._as_real_numeric_array("a", a)
        b_value = exposure_module._as_real_numeric_array("b", b)
        if c is None:
            c_value = np.zeros_like(a_value)
        else:
            c_value = exposure_module._as_real_numeric_array("c", c)

        from . import _core

        result = _core.py_sympson_hetter(
            a_value,
            b_value,
            c_value,
            r_max_value,
            test_length_value,
            n_simulees_value,
            max_iter_value,
            tol_value,
            seed_value,
            q_theta_value,
        )
        return exposure_module.SympsonHetterResult(
            k=np.asarray(result["k"]),
            exposure=np.asarray(result["exposure"]),
            selection=np.asarray(result["selection"]),
            max_exposure=float(result["max_exposure"]),
            n_iter=int(result["n_iter"]),
            converged=bool(result["converged"]),
            history_max_exposure=np.asarray(result["history_max_exposure"]),
        )

    exposure_module.sympson_hetter = safe_sympson_hetter
