"""Synthetic holograms for the reconstruction examples."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from muscopy.dh import MuParameters, make_disk

if TYPE_CHECKING:
    from jax import Array


def disk_phase_object(
    params: MuParameters,
    radius_px: float = 30,
    phase_step_rad: float = 2 * math.pi / 3,
    *,
    center: tuple[int, int] | None = None,
) -> Array:
    """Create a unit-amplitude phase object with a circular phase step.

    Returns
    -------
    Array
        Complex field on the camera grid.
    """
    disk = make_disk(params.img_center if center is None else center, radius_px, params.img_size_px)
    return jnp.exp(1j * phase_step_rad * disk)


def _gaussian(params: MuParameters, sigma_x: float, sigma_y: float) -> Array:
    x, y = jnp.ogrid[: params.img_size_px, : params.img_size_px]
    center_x, center_y = params.img_center
    return jnp.exp(-((x - center_x) ** 2 / (2 * sigma_x**2) + (y - center_y) ** 2 / (2 * sigma_y**2)))


def gaussian_phase_object(params: MuParameters) -> Array:
    """Create a Gaussian phase object whose peak wraps twice at 2π.

    Returns
    -------
    Array
        Complex field on the camera grid.
    """
    return jnp.exp(4j * math.pi * _gaussian(params, 15, 50))


def _apply_pupil(field: Array, pupil: Array) -> Array:
    axes = (-2, -1)
    spectrum = jnp.fft.fftshift(jnp.fft.fft2(field), axes=axes)
    return jnp.fft.ifft2(jnp.fft.ifftshift(spectrum * pupil, axes=axes))


def _unit_power(field: Array) -> Array:
    power = jnp.sum(jnp.abs(field) ** 2, axis=(-2, -1), keepdims=True)
    return field / jnp.sqrt(jnp.where(power > 0, power, 1))


def off_axis_holograms(
    params: MuParameters,
    off_axis_center: tuple[int, int],
    sample_field: Array | None = None,
    *,
    pupil: Array | None = None,
    multiplex_matrix: Array | None = None,
) -> tuple[Array, Array]:
    """Generate off-axis sample holograms and a sample-free reference.

    The default sample is a disk phase object. The centered Fourier pupil
    defaults to the microscope aperture. For multiplexing, pass a stack of
    sample fields and a matrix with one row per recorded hologram and one
    column per sample field. Each field is normalized before mixing.

    Returns
    -------
    tuple[Array, Array]
        Sample intensity image or image stack, and the 2D reference image.
    """
    if sample_field is None:
        sample_field = disk_phase_object(params)
    if pupil is None:
        pupil = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
    sample_field = _unit_power(_apply_pupil(sample_field, pupil))
    if multiplex_matrix is not None:
        sample_field = jnp.einsum("ij,jxy->ixy", multiplex_matrix, sample_field)

    size = params.img_size_px
    x, y = jnp.ogrid[:size, :size]
    center_x, center_y = params.img_center
    carrier = x * (off_axis_center[0] - center_x) / size + y * (off_axis_center[1] - center_y) / size
    reference_beam = _unit_power(jnp.exp(-2j * math.pi * carrier))
    empty_field = _unit_power(jnp.ones_like(reference_beam))
    return jnp.abs(sample_field + reference_beam) ** 2, jnp.abs(empty_field + reference_beam) ** 2


def inline_holograms(params: MuParameters, phase_shifts: Array, ref_amplitude: float) -> tuple[Array, Array]:
    """Generate measurements for phase-shifting inline holography.

    Returns
    -------
    tuple[Array, Array]
        Phase-shifted intensity stack and its ground-truth complex field.
    """
    radius_px = 40
    field = _gaussian(params, radius_px / 2, radius_px / 2) * disk_phase_object(params, radius_px, math.pi / 2)
    pupil = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
    object_field = _apply_pupil(field, pupil)
    reference_beams = ref_amplitude * jnp.exp(1j * phase_shifts[:, None, None])
    return jnp.abs(object_field + reference_beams) ** 2, object_field


def main() -> None:
    """Write a synthetic off-axis hologram pair and its parameters to an NPZ file."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("output", type=Path, help="path to the output .npz file")
    args = parser.parse_args()

    params = MuParameters(na=0.8, wavelength_m=500e-9, img_size_px=512, px_size_m=5e-6 / 60, n_sol=1.33)
    off_axis_center = (100, 100)
    hologram, reference_hologram = off_axis_holograms(params, off_axis_center)
    np.savez(
        args.output,
        hologram=np.asarray(hologram),
        reference_hologram=np.asarray(reference_hologram),
        off_axis_center=off_axis_center,
        **asdict(params),
    )
