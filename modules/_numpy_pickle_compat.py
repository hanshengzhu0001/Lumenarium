"""Load NumPy pickles across NumPy 1.x/2.x module-name changes."""

from __future__ import annotations

import pickle
from typing import Any, BinaryIO


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
