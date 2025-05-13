"""QPI util functions for microscopy data processing."""

import numpy as np
from scipy.fftpack import dct, idct

from muscopy.backend_manager import ArrayProtocol


def unwrap_phase(phase_image: ArrayProtocol) -> ArrayProtocol:
    """Unwraps the phase of a 2D image using the Poisson solver.

    Parameters
    ----------
    phase_image : `ArrayProtocol`
        The wrapped phase image to be unwrapped.

    Returns
    -------
    `ArrayProtocol`
        The unwrapped phase image.
    """
    dx = np.concatenate(
        (
            np.zeros((phase_image.shape[0], 1)),
            _wraptopi(np.diff(phase_image, axis=1, n=1)),
            np.zeros((phase_image.shape[0], 1)),
        ),
        axis=1,
    )
    dy = np.concatenate(
        (
            np.zeros((1, phase_image.shape[1])),
            _wraptopi(np.diff(phase_image, axis=0, n=1)),
            np.zeros((1, phase_image.shape[1])),
        ),
        axis=0,
    )

    rho = np.diff(dx, axis=1, n=1) + np.diff(dy, axis=0, n=1)
    return _solve_poisson(rho)


def _wraptopi(x: ArrayProtocol) -> ArrayProtocol:
    xwrap = np.remainder(x, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = np.remainder(x, np.pi) == 0
    mask3 = np.remainder(x, 2 * np.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * np.pi
    return xwrap


def _solve_poisson(rho: ArrayProtocol) -> ArrayProtocol:
    dct_rho = _dct2(rho)
    n, m = rho.shape
    i, j = np.meshgrid(np.arange(0, n), np.arange(0, m), indexing="ij")
    dct_phi = dct_rho / (2 * (np.cos(np.pi * i / n) + np.cos(np.pi * j / m) - 2))
    dct_phi[0, 0] = 0
    return _idct2(dct_phi)


def _dct2(block: ArrayProtocol) -> ArrayProtocol:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: ArrayProtocol) -> ArrayProtocol:
    return idct(idct(block.T, norm="ortho").T, norm="ortho")
