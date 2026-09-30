# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Host split for explicit bifactor GPU requests.

wgpu 30.0.0 classifies ``DeviceType::Cpu`` as software rendering, which is
the class Mesa lavapipe reports. An explicit bifactor ``device="gpu"`` fit
must use a hardware adapter or raise ``ValueError``. Callers that need a
completed fit treat ``None`` as "this host has no hardware GPU".
``FOCAL_GPU_NATIVE=1`` is the hardware-runner contract: the error is a
failure there, because that job's evidence is the fit itself.

https://docs.rs/wgpu/30.0.0/wgpu/enum.DeviceType.html
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

HARDWARE_GPU_UNAVAILABLE = "no usable hardware GPU path was available"

T = TypeVar("T")


def explicit_gpu_result(thunk: Callable[[], T]) -> T | None:
    """Return ``thunk()``, or ``None`` when this host has no hardware GPU."""
    try:
        return thunk()
    except ValueError as exc:
        if HARDWARE_GPU_UNAVAILABLE not in str(exc):
            raise
        if os.environ.get("FOCAL_GPU_NATIVE") == "1":
            raise AssertionError(
                "hardware GPU runner could not execute the bifactor GPU path"
            ) from exc
        return None


def missing_hardware_bootstrap(exc: BaseException) -> bool:
    """True when every bootstrap replicate failed closed for lack of a GPU.

    A zero-convergence bootstrap raises ``RuntimeError`` whose message
    includes the first replicate error. On the hardware runner that outcome
    is a failed proof, not an acceptable host limitation.
    """
    if not isinstance(exc, RuntimeError):
        return False
    if HARDWARE_GPU_UNAVAILABLE not in str(exc):
        return False
    if os.environ.get("FOCAL_GPU_NATIVE") == "1":
        raise AssertionError(
            "hardware GPU runner recorded no converged bifactor GPU bootstrap replicate"
        ) from exc
    return True
