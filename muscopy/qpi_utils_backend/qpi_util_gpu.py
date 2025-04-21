"""QPI util functions for microscopy data processing."""

import cupy as cp
from cupyx.scipy.fft import dct, idct

from muscopy.backend_manager import ArrayProtocol


def unwrap_phase(phase_image: ArrayProtocol) -> ArrayProtocol:
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
    dx = cp.concatenate(
        (
            cp.zeros((phase_image.shape[0], 1)),
            _wraptopi(cp.diff(phase_image, axis=1, n=1)),
            cp.zeros((phase_image.shape[0], 1)),
        ),
        axis=1,
    )
    dy = cp.concatenate(
        (
            cp.zeros((1, phase_image.shape[1])),
            _wraptopi(cp.diff(phase_image, axis=0, n=1)),
            cp.zeros((1, phase_image.shape[1])),
        ),
        axis=0,
    )

    rho = cp.diff(dx, axis=1, n=1) + cp.diff(dy, axis=0, n=1)
    return _solve_poisson(rho)


def _wraptopi(x: ArrayProtocol) -> ArrayProtocol:
    xwrap = cp.remainder(x, 2 * cp.pi)
    mask = cp.abs(xwrap) > cp.pi
    xwrap[mask] -= 2 * cp.pi * cp.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = cp.remainder(x, cp.pi) == 0
    mask3 = cp.remainder(x, 2 * cp.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * cp.pi
    return xwrap


def _solve_poisson(rho: ArrayProtocol) -> ArrayProtocol:
    dct_rho = _dct2(rho)
    n, m = rho.shape
    i, j = cp.meshgrid(cp.arange(0, n), cp.arange(0, m), indexing="ij")
    dct_phi = dct_rho / (2 * (cp.cos(cp.pi * i / n) + cp.cos(cp.pi * j / m) - 2))
    dct_phi[0, 0] = 0
    return _idct2(dct_phi)


def _dct2(block: ArrayProtocol) -> ArrayProtocol:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: ArrayProtocol) -> ArrayProtocol:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")
