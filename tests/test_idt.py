"""Test cases for IDT module."""

import jax.numpy as jnp
import pytest

from muscopy.cfg import ArrayPrecision
from muscopy.idt import IDTConfig, IDTParameters, compute_idt


def _small_idt_params() -> IDTParameters:
    return IDTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
        num_z_slices=2,
    )


def test_idt_config_defaults_to_32_bit_precision() -> None:
    config = IDTConfig()

    assert config.precision.int_precision() == "int32"
    assert config.precision.float_precision() == "float32"
    assert config.precision.complex_precision() == "complex64"


def test_compute_idt_returns_32_bit_arrays_by_default() -> None:
    params = _small_idt_params()
    intensity_images = [jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)]
    ref_intensity_images = [jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)]
    u_illumination_list = [(0.0, 0.0)]

    n_re, n_im = compute_idt(params, intensity_images, ref_intensity_images, u_illumination_list)

    expected_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1, params.num_z_slices)
    assert n_re.shape == expected_shape
    assert n_im.shape == expected_shape
    assert n_re.dtype == jnp.float32
    assert n_im.dtype == jnp.float32
    assert bool(jnp.allclose(n_re, 0.0))
    assert bool(jnp.allclose(n_im, 0.0))


def test_compute_idt_rejects_mutated_64_bit_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("muscopy.cfg._jax_x64_enabled", lambda: False)
    params = _small_idt_params()
    precision = ArrayPrecision()
    precision.float_length = 64
    config = IDTConfig(precision=precision)

    with pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"):
        compute_idt(params, [], [], [], config=config)
