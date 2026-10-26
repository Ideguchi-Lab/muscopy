"""Test cases for phase noise module."""

import jax.numpy as jnp
import pytest
from jax import Array

from muscopy.dh import MuParameters
from muscopy.phase_noise import calc_phase_noise, calc_visibility


def _make_hologram(params: MuParameters) -> tuple[Array, tuple[int, int]]:
    height, width = params.img_shape_px
    xx, yy = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")
    shift = (3 * height // 16, 3 * width // 16)
    reference_wave = jnp.exp(-2j * jnp.pi * (shift[0] * xx / height + shift[1] * yy / width))
    hologram = jnp.abs(1 + reference_wave) ** 2
    return hologram, (params.img_center[0] + shift[0], params.img_center[1] + shift[1])


@pytest.mark.parametrize("img_size_px", [64, (64, 32)])
def test_calc_visibility_of_ideal_fringe_is_one(img_size_px: int | tuple[int, int]) -> None:
    params = MuParameters(na=0.125, wavelength_m=1.0, img_size_px=img_size_px, px_size_m=1.0, n_sol=1.33)
    hologram, offaxis_center = _make_hologram(params)

    visibility = calc_visibility(hologram, params, offaxis_center)

    assert visibility.shape == params.aperturesize_px_xy
    assert bool(jnp.allclose(visibility, 1.0, atol=1e-5))


def test_calc_phase_noise_square_tuple_matches_int() -> None:
    params_int = MuParameters(na=0.125, wavelength_m=1.0, img_size_px=64, px_size_m=1.0, n_sol=1.33)
    params_tuple = MuParameters(na=0.125, wavelength_m=1.0, img_size_px=(64, 64), px_size_m=1.0, n_sol=1.33)
    hologram, offaxis_center = _make_hologram(params_int)

    noise_int = calc_phase_noise(hologram, params_int, offaxis_center, fullwell=10000, bit_depth=12)
    noise_tuple = calc_phase_noise(hologram, params_tuple, offaxis_center, fullwell=10000, bit_depth=12)

    assert bool(jnp.all(jnp.isfinite(noise_int)))
    assert bool(jnp.array_equal(noise_int, noise_tuple))
