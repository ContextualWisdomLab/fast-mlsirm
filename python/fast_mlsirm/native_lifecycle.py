"""Native mask-GRM lifecycle transport.

GPU requests retain host EAP, probability-table and information phases.
Private fitted state contains responses/scores; store it with restricted access.
It is trusted local persistence, not authenticated evidence or an EM checkpoint.
"""
from . import _core

capabilities = _core.sequential_dif_capabilities
complete_gpu = _core.sequential_dif_complete_gpu
round_gpu = _core.sequential_dif_round_gpu
fit_gpu = _core.sequential_dif_fit_gpu
fit_private = _core.sequential_dif_fit_private
restore_summary = _core.sequential_dif_restore_summary
roundtrip_private = _core.sequential_dif_roundtrip_private
