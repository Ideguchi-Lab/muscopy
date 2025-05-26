"""QPI util functions for microscopy data processing.

This module provides:

- `unwrap_phase`: Unwraps the phase of a 2D image using the Poisson solver.
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array


def unwrap_phase(phase_image: Array) -> Array:
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
    dx = jnp.concatenate(
        (
            jnp.zeros((phase_image.shape[0], 1)),
            _wraptopi(jnp.diff(phase_image, axis=1, n=1)),
            jnp.zeros((phase_image.shape[0], 1)),
        ),
        axis=1,
    )
    dy = jnp.concatenate(
        (
            jnp.zeros((1, phase_image.shape[1])),
            _wraptopi(jnp.diff(phase_image, axis=0, n=1)),
            jnp.zeros((1, phase_image.shape[1])),
        ),
        axis=0,
    )

    rho = jnp.diff(dx, axis=1, n=1) + jnp.diff(dy, axis=0, n=1)
    return _solve_poisson(rho)


def _wraptopi(x: Array) -> Array:
    xwrap = jnp.remainder(x, 2 * jnp.pi)
    mask = jnp.abs(xwrap) > jnp.pi
    xwrap = jnp.where(mask, xwrap - 2 * jnp.pi * jnp.sign(xwrap), xwrap)
    mask1 = x < 0
    mask2 = jnp.remainder(x, jnp.pi) == 0
    mask3 = jnp.remainder(x, 2 * jnp.pi) != 0
    return jnp.where(mask1 & mask2 & mask3, xwrap - 2 * jnp.pi, xwrap)


def _solve_poisson(rho: Array) -> Array:
    dct_rho = _dct2(rho)
    n, m = rho.shape
    i, j = jnp.meshgrid(jnp.arange(0, n), jnp.arange(0, m), indexing="ij")
    dct_phi = dct_rho / (2 * (jnp.cos(jnp.pi * i / n) + jnp.cos(jnp.pi * j / m) - 2))
    dct_phi = dct_phi.at[0, 0].set(0)
    return _idct2(dct_phi)


def _dct2(block: Array) -> Array:
    return jsp.fft.dct(jsp.fft.dct(block.T, norm="ortho").T, norm="ortho")


def _idct2(block: Array) -> Array:
    return jsp.fft.idct(jsp.fft.idct(block.T, norm="ortho").T, norm="ortho")
