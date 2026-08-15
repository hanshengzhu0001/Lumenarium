"""Compatibility loader for NumPy pickles across NumPy 1.x and 2.x.

NumPy 2 serializes some classes under ``numpy._core`` while Blender 4.3's
bundled Python environment may expose the same classes under ``numpy.core``.
Only the module prefix is translated; the pickle payload is otherwise loaded
with the standard unpickler.
"""

from __future__ import annotations

import pickle
from typing import BinaryIO, Any


def remap_numpy_pickle_module(module: str) -> str:
    if module == "numpy._core":
        return "numpy.core"
    if module.startswith("numpy._core."):
        return "numpy.core." + module[len("numpy._core.") :]
    return module


class NumPyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(remap_numpy_pickle_module(module), name)


def load_numpy_compatible_pickle(stream: BinaryIO) -> Any:
    return NumPyCompatUnpickler(stream).load()
