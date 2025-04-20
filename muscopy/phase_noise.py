"""Calculation module for phase noise in digital holography.

This module provides:

- `calc_visibility`: Calculate the visibility of a hologram.
- `calc_phase_noise`: Calculate the theoretical phase noise in a given hologram.
"""

import math
import types

from muscopy.backend_manager import ArrayProtocol
from muscopy.qpi import QPIParameters, crop_array, make_disk


def _get_dc_ac(
    backend: types.ModuleType,
    hologram: ArrayProtocol,
    params: QPIParameters,
    offaxis_center: tuple[int, int],
) -> tuple[ArrayProtocol, ArrayProtocol]:
    scale_factor = params.aperturesize_px / params.img_size_px
    fft = backend.fft.fftshift(backend.fft.fft2(hologram))
    dc_disk = make_disk(backend, params.img_center, params.aperturesize_px // 2, params.img_size_px)
    ac_disk = make_disk(backend, offaxis_center, params.aperturesize_px // 2, params.img_size_px)

    dc_fft = fft * dc_disk
    ac_fft = fft * ac_disk

    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize_px)
    ac_cropped = crop_array(ac_fft, offaxis_center, params.aperturesize_px)

    dc = backend.fft.ifft2(backend.fft.ifftshift(dc_cropped)) * scale_factor**2
    ac = backend.fft.ifft2(backend.fft.ifftshift(ac_cropped)) * scale_factor**2

    return dc, ac


def _get_visibility(
    backend: types.ModuleType,
    dc: ArrayProtocol,
    ac: ArrayProtocol,
) -> ArrayProtocol:
    return backend.abs(ac) / backend.abs(dc) * 2


def _get_phase_noise(  # noqa: PLR0913, PLR0917
    backend: types.ModuleType,
    visibility: ArrayProtocol,
    aperturesize: int,
    dc_intensity: ArrayProtocol,
    sensorsize: int,
    sensor_noise: int = 0,
) -> ArrayProtocol:
    if visibility.shape != dc_intensity.shape:
        msg = "Visibility and DC intensity must have the same shape"
        raise ValueError(msg)
    aperture_area = math.pi * (aperturesize / 2) ** 2
    sensor_area = sensorsize**2
    return backend.sqrt(
        2 * aperture_area * (dc_intensity + sensor_noise**2) / (visibility**2 * dc_intensity**2 * sensor_area)
    )


def calc_visibility(
    backend: types.ModuleType,
    hologram: ArrayProtocol,
    params: QPIParameters,
    offaxis_center: tuple[int, int],
) -> ArrayProtocol:
    r"""Calculate the visibility of a hologram.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    hologram : `ArrayProtocol`
        The hologram array
    params : `QPIParameters`
        QPI parameters
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of the off-axis digital holography

    Returns
    -------
    `ArrayProtocol`
        The visibility of the hologram
    """
    dc, ac = _get_dc_ac(backend, hologram, params, offaxis_center)
    return _get_visibility(backend, dc, ac)


def calc_phase_noise(
    backend: types.ModuleType,
    hologram: ArrayProtocol,
    params: QPIParameters,
    offaxis_center: tuple[int, int],
    fullwell: int,
    bit_depth: int,
    sensor_noise: int = 0,
) -> ArrayProtocol:
    r"""Calculate the theoretical phase noise in a given hologram.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    hologram : `ArrayProtocol`
        The hologram array
    params : `QPIParameters`
        QPI parameters
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
    `ArrayProtocol`
        The theoretical phase noise of the hologram
    """
    dc, ac = _get_dc_ac(backend, hologram, params, offaxis_center)
    visibility = _get_visibility(backend, dc, ac)
    dc_factor = fullwell / (2**bit_depth)
    return _get_phase_noise(
        backend,
        visibility,
        params.aperturesize_px,
        dc_factor * backend.abs(dc),
        params.img_size_px,
        sensor_noise,
    )
