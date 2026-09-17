"""Load the interaction-map envelope PyO3 module from the package shared library."""

from __future__ import annotations

from importlib.machinery import ExtensionFileLoader
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
import sys
import threading


_MODULE_NAME = "fast_mlsirm._interaction_map_core"
_LOAD_LOCK = threading.Lock()


def interaction_map_core() -> ModuleType:
    """Return the cached secondary Rust interaction-map extension module.

    Concurrent callers are serialized through initialization so no caller can
    observe the temporary ``sys.modules`` entry before ``exec_module`` has
    completed.

    Raises
    ------
    ImportError
        If the installed wheel predates the interaction-map envelope entrypoint
        or the extension loader cannot initialize ``PyInit__interaction_map_core``.
    """

    with _LOAD_LOCK:
        cached = sys.modules.get(_MODULE_NAME)
        if cached is not None:
            return cached

        from . import _core

        binary_path = getattr(_core, "__file__", None)
        if not binary_path:
            raise ImportError("fast_mlsirm._core does not expose an extension path")
        loader = ExtensionFileLoader(_MODULE_NAME, binary_path)
        spec = spec_from_loader(_MODULE_NAME, loader)
        if spec is None:
            raise ImportError(
                "could not create the fast_mlsirm interaction-map extension spec"
            )
        module = module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_MODULE_NAME, None)
            raise
        return module
