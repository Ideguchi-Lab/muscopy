"""Test cases for QPI module."""

import jax.numpy as jnp
import pytest
from jax import Array

from muscopy.dh import MuParameters
from muscopy.qpi import mip_qpi, qpi

# The aperture radius is 1/8 of each image axis and the off-axis carrier sits at 3/16 of it,
# so the cropped ellipse excludes the DC peak and the twin image on every axis.
NA = 0.125


def _make_params(img_size_px: int | tuple[int, int]) -> MuParameters:
    return MuParameters(na=NA, wavelength_m=1.0, img_size_px=img_size_px, px_size_m=1.0, n_sol=1.33)


def _sample_phase(params: MuParameters, xx: Array, yy: Array) -> Array:
    # Two cosine cycles along each axis keep every Jacobi-Anger harmonic that matters inside the pupil.
    height, width = params.img_shape_px
    return 0.5 * jnp.cos(2 * jnp.pi * (2 * xx / height + 2 * yy / width))


def _make_holograms(params: MuParameters, phase_scale: float = 1.0) -> tuple[Array, Array, tuple[int, int]]:
    height, width = params.img_shape_px
    xx, yy = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    shift = (3 * height // 16, 3 * width // 16)
    offaxis_center = (params.img_center[0] + shift[0], params.img_center[1] + shift[1])
    reference_wave = jnp.exp(-2j * jnp.pi * (shift[0] * xx / height + shift[1] * yy / width))
    sample_wave = jnp.exp(1j * phase_scale * _sample_phase(params, xx, yy))
    hologram = jnp.abs(sample_wave + reference_wave) ** 2
    reference_hologram = jnp.abs(1 + reference_wave) ** 2
    return hologram, reference_hologram, offaxis_center


def _expected_phase(params: MuParameters) -> Array:
    # The cropped spectrum resamples the band-limited field on a grid of aperturesize_px_xy points.
    height, width = params.img_shape_px
    size_x, size_y = params.aperturesize_px_xy
    xx, yy = jnp.meshgrid(jnp.arange(size_x) * height / size_x, jnp.arange(size_y) * width / size_y, indexing="ij")
    return _sample_phase(params, xx, yy)


@pytest.mark.parametrize("img_size_px", [128, (256, 128), (128, 256)])
def test_qpi_reconstructs_phase_with_elliptical_crop(img_size_px: int | tuple[int, int]) -> None:
    params = _make_params(img_size_px)
    hologram, reference_hologram, offaxis_center = _make_holograms(params)

    phase = qpi(hologram, reference_hologram, params, offaxis_center)

    assert phase.shape == params.aperturesize_px_xy
    assert bool(jnp.allclose(phase, _expected_phase(params), atol=1e-3))


def test_qpi_rectangular_image_returns_one_phase_per_center() -> None:
    params = _make_params((256, 128))
    hologram, reference_hologram, offaxis_center = _make_holograms(params)

    phases = qpi(hologram, reference_hologram, params, [offaxis_center, offaxis_center])

    assert len(phases) == 2
    for phase in phases:
        assert phase.shape == params.aperturesize_px_xy


def test_mip_qpi_rectangular_image_recovers_phase_difference() -> None:
    params = _make_params((256, 128))
    hologram_on, _, offaxis_center = _make_holograms(params, phase_scale=1.2)
    hologram_off, _, _ = _make_holograms(params, phase_scale=1.0)

    phase = mip_qpi(hologram_on, hologram_off, params, offaxis_center)

    assert phase.shape == params.aperturesize_px_xy
    assert bool(jnp.allclose(phase, 0.2 * _expected_phase(params), atol=1e-3))
