"""QPI util functions for microscopy data processing.

This module provides:

- `unwrap_phase`: Unwraps the phase of a 2D image using the Poisson solver.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array


def unwrap_phase(phase_image: Array, *, roi: Array | None = None, keep_mean: bool = True) -> Array:
    """Unwraps the phase of a 2D image using the Poisson solver.

    Parameters
    ----------
    phase_image : `Array`
        The wrapped phase image to be unwrapped.

    Returns
    -------
    `Array`
        The unwrapped phase image.
    """
    if roi is None:
        roi = jnp.ones_like(phase_image)

    wrapped = jnp.where(roi, phase_image, 0.0)

    dx = jnp.zeros_like(wrapped)
    dx = dx.at[:, 1:].set(_wraptopi(jnp.diff(wrapped, axis=1)))
    dy = jnp.zeros_like(wrapped)
    dy = dy.at[1:, :].set(_wraptopi(jnp.diff(wrapped, axis=0)))
    rho = jnp.diff(dx, axis=1, prepend=0.0) + jnp.diff(dy, axis=0, prepend=0.0)

    rho = jnp.where(roi, rho, 0.0)

    phi = _solve_poisson(rho)
    phi = jnp.where(roi, phi, phase_image)

    if keep_mean:
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
