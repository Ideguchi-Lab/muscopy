"""CuPy implementation of hologram visibility."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp

from muscopy.cuda.dh import get_spectrums

if TYPE_CHECKING:
    from muscopy.dh import MuParameters

__all__ = ["calc_visibility"]


def calc_visibility(
    hologram: cp.ndarray,
    params: MuParameters,
    offaxis_center: tuple[int, int],
) -> cp.ndarray:
    r"""Calculate the fringe visibility of a hologram using CuPy.

    Parameters
    ----------
    hologram : CuPy array
        Square hologram of size ``params.img_size_px``.
    params : `MuParameters`
        Microscopy parameters.
    offaxis_center : `tuple`\[`int`, `int`\]
        Row and column of the sideband in the shifted Fourier spectrum.

    Returns
    -------
    CuPy array
        Visibility :math:`2|AC|/|DC|` on the cropped aperture grid.
    """
    spectrum = cp.fft.fftshift(cp.fft.fft2(hologram))
    dc_spectrum, ac_spectrum = get_spectrums(spectrum, params, (params.img_center, offaxis_center))
    dc = cp.fft.ifft2(cp.fft.ifftshift(dc_spectrum))
    ac = cp.fft.ifft2(cp.fft.ifftshift(ac_spectrum))
    # The DC and AC components have the same normalization, which cancels in the ratio.
    return 2 * (cp.abs(ac) / cp.abs(dc))
