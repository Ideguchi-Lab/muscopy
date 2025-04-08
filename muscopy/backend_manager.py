"""Backend manager for Muscopy."""

import types
from importlib.util import find_spec

import typing_extensions


class BackendManager:
    """Backend manager for Muscopy."""

    def __init__(self) -> None:
        self.__backend = "numpy"

    @property
    def backend(self) -> str:
        """Get the current backend."""
        return self.__backend

    def use_numpy(self) -> None:
        """Set the backend to numpy."""
        self.__backend = "numpy"

    def use_cupy(self) -> None:
        """Set the backend to cupy.

        Raises
        ------
        ImportError
            If cupy is not available.
        """
        if not bool(find_spec("cupy")):
            msg = "Cupy is not available. Please install cupy to use the GPU backend."
            raise ImportError(msg)
        self.__backend = "cupy"

    def get_backend(self) -> types.ModuleType:
        """Get the current backend module.

        Returns
        -------
        types.ModuleType
            The current backend module (numpy or cupy).
        """
        if self.__backend == "numpy":
            import numpy as np  # noqa: PLC0415

            return np
        if self.__backend == "cupy":
            import cupy as cp  # noqa: PLC0415

            return cp
        typing_extensions.assert_never(self.__backend)
