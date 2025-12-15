"""QPI util functions for microscopy data processing.

This module provides:

- `unwrap_phase`: Unwraps the phase of a 2D image using the Poisson solver.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import Array
from skimage.restoration import unwrap_phase as skimage_unwrap_phase


def unwrap_phase(
    phase_image: Array, *, roi: Array | None = None, keep_mean: bool = True, use_skimage: bool = False
) -> Array:
    """Unwraps the phase of a 2D image using the Poisson solver.

    Parameters
    ----------
    phase_image : `Array`
        The wrapped phase image to be unwrapped.
    roi : `Array`, optional
        A region of interest mask where the unwrapping should be applied. If `None`,
        the entire image is considered. Default is `None`.
    keep_mean : `bool`, optional
        If `True`, the mean of the original phase image is added back to the unwrapped phase.
        Default is `True`.
    use_skimage : `bool`, optional
        If `True`, uses `skimage.restoration.unwrap_phase` for unwrapping. If `False`, uses the Poisson solver method.
        Default is `False`.

    Returns
    -------
    `Array`
        The unwrapped phase image.
    """
    if use_skimage:
        # move CPU if necessary
        phase_cpu = np.asarray(phase_image)
        unwrapped_cpu = skimage_unwrap_phase(phase_cpu)
        return jnp.asarray(unwrapped_cpu)

    original_roi = roi
    roi = jnp.ones(phase_image.shape, dtype=bool) if roi is None else roi.astype(bool)

    # Calculate forward differences (shape: (n, m-1), (n-1, m))
    dx = _wraptopi(jnp.diff(phase_image, axis=1))
    dy = _wraptopi(jnp.diff(phase_image, axis=0))

    # Mask gradients: only use differences where both endpoints are inside ROI
    roi_x = roi[:, 1:] & roi[:, :-1]
    roi_y = roi[1:, :] & roi[:-1, :]

    dx = jnp.where(roi_x, dx, 0.0)
    dy = jnp.where(roi_y, dy, 0.0)

    # Divergence with Neumann BC: pad both sides with 0 via prepend+append
    rho = jnp.diff(dx, axis=1, prepend=0.0, append=0.0) + jnp.diff(dy, axis=0, prepend=0.0, append=0.0)

    rho = jnp.where(roi, rho, 0.0)

    phi = _solve_poisson(rho)
    phi = jnp.where(roi, phi, phase_image)

    if keep_mean:
        if original_roi is not None:
            # Use ROI-masked mean for ROI case, only apply to ROI pixels
            mean_roi = jnp.sum(jnp.where(roi, phase_image, 0.0)) / jnp.sum(roi)
            phi = jnp.where(roi, phi + mean_roi, phi)
        else:
            phi = phi + phase_image.mean()  # noqa: PLR6104

    return phi


def _wraptopi(x: Array) -> Array:
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _solve_poisson(rho: Array) -> Array:
    dct_rho = _dct2(rho)
    n, m = rho.shape
    i, j = jnp.meshgrid(jnp.arange(0, n), jnp.arange(0, m), indexing="ij")

    denom = 2.0 * (jnp.cos(jnp.pi * i / n) + jnp.cos(jnp.pi * j / m) - 2.0)
    denom_safe = jnp.where((i == 0) & (j == 0), 1.0, denom)

    dct_phi = dct_rho / denom_safe
    dct_phi = dct_phi.at[0, 0].set(0.0)
    return _idct2(dct_phi)


def _dct2(x: Array) -> Array:
    return jsp.fft.dct(jsp.fft.dct(x.T, type=2, norm="ortho").T, type=2, norm="ortho")


def _idct2(x: Array) -> Array:
    return jsp.fft.idct(jsp.fft.idct(x.T, type=2, norm="ortho").T, type=2, norm="ortho")
