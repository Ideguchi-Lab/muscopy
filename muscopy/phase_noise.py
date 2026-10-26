"""Calculation module for phase noise in digital holography.

This module provides:

- `calc_visibility`: Calculate the visibility of a hologram.
- `calc_phase_noise`: Calculate the theoretical phase noise in a given hologram.
"""

import math

import jax.numpy as jnp
from jax import Array

from muscopy.dh import MuParameters, crop_array, make_ellipse

__all__ = [
    "calc_phase_noise",
    "calc_visibility",
]


def _get_dc_ac(
    hologram: Array,
    params: MuParameters,
    offaxis_center: tuple[int, int],
) -> tuple[Array, Array]:
    aperture_x, aperture_y = params.aperturesize_px_xy
    img_x, img_y = params.img_shape_px
    scale_factor_sq = (aperture_x / img_x) * (aperture_y / img_y)
    fft = jnp.fft.fftshift(jnp.fft.fft2(hologram))
    dc_disk = make_ellipse(params.img_center, params.aperture_radius_px_xy, params.img_shape_px)
    ac_disk = make_ellipse(offaxis_center, params.aperture_radius_px_xy, params.img_shape_px)

    dc_fft = fft * dc_disk
    ac_fft = fft * ac_disk

    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize_px_xy)
    ac_cropped = crop_array(ac_fft, offaxis_center, params.aperturesize_px_xy)

    dc = jnp.fft.ifft2(jnp.fft.ifftshift(dc_cropped)) * scale_factor_sq
    ac = jnp.fft.ifft2(jnp.fft.ifftshift(ac_cropped)) * scale_factor_sq

    return dc, ac


def _get_visibility(
    dc: Array,
    ac: Array,
) -> Array:
    return jnp.abs(ac) / jnp.abs(dc) * 2


def _get_phase_noise(
    visibility: Array,
    aperturesize: tuple[int, int],
    dc_intensity: Array,
    sensorsize: tuple[int, int],
    sensor_noise: int = 0,
) -> Array:
    if visibility.shape != dc_intensity.shape:
        msg = "Visibility and DC intensity must have the same shape"
        raise ValueError(msg)
    aperture_area = math.pi * (aperturesize[0] / 2) * (aperturesize[1] / 2)
    sensor_area = sensorsize[0] * sensorsize[1]
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


def calc_phase_noise(
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
        params.aperturesize_px_xy,
        dc_factor * jnp.abs(dc),
        params.img_shape_px,
        sensor_noise,
    )
