"""Synthetic tomographic measurements for examples using the optional MLB simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from jax import Array
    from muscopy_mlbsim import MLBParameters  # pyright: ignore[reportMissingImports]

    from muscopy.odt import ODTParameters


def _sphere_refractive_index(params: MLBParameters, radius_um: float, delta_n: float) -> Array:
    shape = (params.num_layers, *params.xy_shape)
    z, y, x = jnp.ogrid[: shape[0], : shape[1], : shape[2]]
    radius_px = int(radius_um * 1e-6 / params.dxy_m)
    in_sphere = (z - shape[0] // 2) ** 2 + (y - shape[1] // 2) ** 2 + (x - shape[2] // 2) ** 2 < radius_px**2
    return jnp.where(in_sphere, params.n_medium + delta_n, params.n_medium).astype(jnp.float32)


def _detector_intensity(field: Array, image_shape: tuple[int, int]) -> Array:
    """Pad the field spectrum to the camera size and retain relative intensity across frames.

    Returns
    -------
    Array
        Detector intensity image.

    Raises
    ------
    ValueError
        If the camera grid is smaller than the detector field.
    """
    height, width = field.shape
    if image_shape[0] < height or image_shape[1] < width:
        msg = "image_shape must be at least as large as the MLB lateral field shape."
        raise ValueError(msg)

    field_fft = jnp.fft.fftshift(jnp.fft.fft2(field))
    image_fft = jnp.zeros(image_shape, dtype=field_fft.dtype)
    row = image_shape[0] // 2 - height // 2
    col = image_shape[1] // 2 - width // 2
    image_fft = image_fft.at[row : row + height, col : col + width].set(field_fft)
    image_field = jnp.fft.ifft2(jnp.fft.ifftshift(image_fft))
    return jnp.asarray(jnp.abs(image_field) ** 2, dtype=jnp.float32)


def simulate_mlb_measurements(
    params: ODTParameters,
    *,
    radius_um: float,
    delta_n: float,
    num_angles: int,
    off_axis_center: tuple[int, int] | None = None,
) -> tuple[list[Array], list[Array], list[tuple[float, float]]]:
    """Simulate sphere and background measurements on a ring of illumination angles.

    With an off-axis center, record holograms for ODT. With the default
    center of None, record detector intensities for IDT.

    Returns
    -------
    tuple[list[Array], list[Array], list[tuple[float, float]]]
        Sample images, background images, and illumination frequencies in pixels,
        all in acquisition order.
    """
    # Import the simulator only when generating MLB measurements.
    from muscopy_mlbsim import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        HologramGenerator,
        MLBForward,
        MLBParameters,
        get_oblique_wave_fft,
        get_scatter_potential,
    )

    mlb_params = MLBParameters(
        wavelength_m=params.wavelength_m,
        numerical_aperture=params.na,
        n_medium=params.n_sol,
        xy_shape=(2 * params.aperturesize_px, 2 * params.aperturesize_px),
        num_layers=params.freq_axial_extent_px,
        dxy_m=params.imgpx_lateral_m_per_px,
        dz_m=params.imgpx_axial_m_per_px,
    )
    scattering_potential = get_scatter_potential(mlb_params, _sphere_refractive_index(mlb_params, radius_um, delta_n))
    forward = MLBForward(mlb_params)
    hologram_generator = None
    if off_axis_center is not None:
        hologram_generator = HologramGenerator(mlb_params)
        hologram_generator.set_offaxis_position(*off_axis_center)

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    illumination_radius = params.light_freq_px * params.na_illumination / params.n_sol
    illumination = [
        (float(illumination_radius * np.cos(angle)), float(illumination_radius * np.sin(angle))) for angle in angles
    ]
    image_shape = (params.img_size_px, params.img_size_px)
    target_images: list[Array] = []
    reference_images: list[Array] = []

    for ux, uy in tqdm(illumination, desc="Simulating MLB measurements"):
        input_field = get_oblique_wave_fft(mlb_params, ux * params.k_per_px, uy * params.k_per_px)
        target_field = forward.simulate_forward_detector_field(input_field, scattering_potential)
        reference_field = forward.propagate_forward_to_detector(input_field)
        for field, images in ((target_field, target_images), (reference_field, reference_images)):
            if hologram_generator is None:
                images.append(_detector_intensity(field, image_shape))
            else:
                hologram_generator.set_target_field(field)
                images.append(hologram_generator.generate_hologram(image_shape, bit_depth=16, output_dtype="float32"))

    return target_images, reference_images, illumination
