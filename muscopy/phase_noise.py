"""Calculation module for phase noise in digital holography.

This module provides:

- `calc_visibility`: Calculate the visibility of a hologram.
- `calc_phase_noise`: Calculate the theoretical phase noise in a given hologram.
"""

import math

import jax.numpy as jnp
from jax import Array

from muscopy.dh import MuParameters, crop_array, make_disk


def _get_dc_ac(
    hologram: Array,
    params: MuParameters,
    offaxis_center: tuple[int, int],
) -> tuple[Array, Array]:
    scale_factor = params.aperturesize_px / params.img_size_px
    fft = jnp.fft.fftshift(jnp.fft.fft2(hologram))
    dc_disk = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
    ac_disk = make_disk(offaxis_center, params.aperturesize_px // 2, params.img_size_px)

    dc_fft = fft * dc_disk
    ac_fft = fft * ac_disk

    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize_px)
    ac_cropped = crop_array(ac_fft, offaxis_center, params.aperturesize_px)

    dc = jnp.fft.ifft2(jnp.fft.ifftshift(dc_cropped)) * scale_factor**2
    ac = jnp.fft.ifft2(jnp.fft.ifftshift(ac_cropped)) * scale_factor**2

    return dc, ac


def _get_visibility(
    dc: Array,
    ac: Array,
) -> Array:
    return jnp.abs(ac) / jnp.abs(dc) * 2


def _get_phase_noise(
    visibility: Array,
    aperturesize: int,
    dc_intensity: Array,
    sensorsize: int,
    sensor_noise: int = 0,
) -> Array:
    if visibility.shape != dc_intensity.shape:
        msg = "Visibility and DC intensity must have the same shape"
        raise ValueError(msg)
    aperture_area = math.pi * (aperturesize / 2) ** 2
    sensor_area = sensorsize**2
    return jnp.sqrt(
        2 * aperture_area * (dc_intensity + sensor_noise**2) / (visibility**2 * dc_intensity**2 * sensor_area)
    )


def calc_visibility(
    hologram: Array,
    params: MuParameters,
    offaxis_center: tuple[int, int],
) -> Array:
    r"""Calculate the visibility of a hologram.

    Parameters
    ----------
    hologram : `Array`
        The hologram array
    params : `MuParameters`
        Microscopy parameters
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of the off-axis digital holography

    Returns
    -------
    `Array`
        The visibility of the hologram
    """
    dc, ac = _get_dc_ac(hologram, params, offaxis_center)
    return _get_visibility(dc, ac)


def calc_phase_noise(  # noqa: PLR0913, PLR0917
    hologram: Array,
    params: MuParameters,
    offaxis_center: tuple[int, int],
    fullwell: int,
    bit_depth: int,
    sensor_noise: int = 0,
) -> Array:
    r"""Calculate the theoretical phase noise in a given hologram.

    Parameters
    ----------
    hologram : `Array`
        The hologram array
    params : `MuParameters`
        Microscopy parameters
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of the off-axis digital holography
    fullwell : `int`
        The fullwell capacity of the image sensor
    bit_depth : `int`
        The bit depth of the image sensor
    sensor_noise : `int`, optional
        The sensor noise of the image sensor, by default 0

    Returns
    -------
    `Array`
        The theoretical phase noise of the hologram
    """
    dc, ac = _get_dc_ac(hologram, params, offaxis_center)
    visibility = _get_visibility(dc, ac)
    dc_factor = fullwell / (2**bit_depth)
    return _get_phase_noise(
        visibility,
        params.aperturesize_px,
        dc_factor * jnp.abs(dc),
        params.img_size_px,
        sensor_noise,
    )
