<<<<<<< HEAD
"""QPI util functions for microscopy data processing."""
=======
"""QPI util functions for microscopy data processing.

This module provides:

- `unwrap_phase`: Unwraps the phase of a 2D image using the Poisson solver.
"""
>>>>>>> df775b41a210472cb388a6c9e441931b0891d6a4

from muscopy import qpi_utils_backend
from muscopy.backend_manager import ArrayProtocol, BackendManager


def unwrap_phase(bmg: BackendManager, phase_image: ArrayProtocol) -> ArrayProtocol:
    """Unwraps the phase of a 2D image using the Poisson solver.

    Parameters
    ----------
<<<<<<< HEAD
    phase_image : ArrayProtocol
=======
    bmg : `BackendManager`
        The backend manager to use for the operation.
    phase_image : `ArrayProtocol`
>>>>>>> df775b41a210472cb388a6c9e441931b0891d6a4
        The wrapped phase image to be unwrapped.

    Returns
    -------
<<<<<<< HEAD
    ArrayProtocol
=======
    `ArrayProtocol`
>>>>>>> df775b41a210472cb388a6c9e441931b0891d6a4
        The unwrapped phase image.
    """
    if bmg.backend == "numpy":
        return qpi_utils_backend.qpi_util_cpu.unwrap_phase(phase_image)
    if bmg.backend == "cupy":
        return qpi_utils_backend.qpi_util_gpu.unwrap_phase(phase_image)
    msg = f"Backend {bmg.backend} is not supported for unwrap_phase."
    raise NotImplementedError(msg)
