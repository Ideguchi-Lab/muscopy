"""QPI util functions for microscopy data processing."""

from muscopy import qpi_utils_backend
from muscopy.backend_manager import ArrayProtocol, BackendManager


def unwrap_phase(bmg: BackendManager, phase_image: ArrayProtocol) -> ArrayProtocol:
    """Unwraps the phase of a 2D image using the Poisson solver.

    Parameters
    ----------
    phase_image : ArrayProtocol
        The wrapped phase image to be unwrapped.

    Returns
    -------
    ArrayProtocol
        The unwrapped phase image.
    """
    if bmg.backend == "numpy":
        return qpi_utils_backend.qpi_util_cpu.unwrap_phase(phase_image)
    if bmg.backend == "cupy":
        return qpi_utils_backend.qpi_util_gpu.unwrap_phase(phase_image)
    msg = f"Backend {bmg.backend} is not supported for unwrap_phase."
    raise NotImplementedError(msg)
