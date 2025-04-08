"""Backend manager for Muscopy."""

from importlib.util import find_spec


class BackendManager:
    """Backend manager for Muscopy."""

    def __init__(self) -> None:
        self._backend = "numpy"

    @property
    def backend(self) -> str:
        """Get the current backend."""
        return self._backend

    def use_numpy(self) -> None:
        """Set the backend to numpy."""
        self._backend = "numpy"

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
        self._backend = "cupy"
