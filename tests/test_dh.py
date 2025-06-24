import math
from dataclasses import fields

import jax.numpy as jnp
import pytest

from muscopy.dh import (
    MuParameters,
    correct_offset,
    correct_offset_idh,
    crop_array,
    get_spectrum,
    get_spectrums,
    inline_dh,
    make_disk,
    meshgrid_freq,
    print_all_parameters,
    propagate_fresnel,
    support_constraint,
)


def test_mu_parameters_properties() -> None:
    # Set fixed parameters
    na = 1.4
    wavelength_m = 550e-9
    img_size_px = 256
    px_size_m = 6.5e-6
    n_sol = 1.33
    params = MuParameters(na, wavelength_m, img_size_px, px_size_m, n_sol)

    # Test img_center property
    expected_img_center = (img_size_px // 2, img_size_px // 2)
    assert params.img_center == expected_img_center

    # Test freq_per_px property
    expected_freq_per_px = 1 / (px_size_m * img_size_px)
    assert abs(params.freq_per_px - expected_freq_per_px) < 1e-12

    # Test k_per_px property
    expected_k_per_px = 2 * math.pi * expected_freq_per_px
    assert abs(params.k_per_px - expected_k_per_px) < 1e-12

    # Test aperturesize_px property
    expected_aperture = 2 * round(na / wavelength_m / expected_freq_per_px) + 1
    assert params.aperturesize_px == expected_aperture

    # Test light_freq_px property
    expected_light_freq_px = n_sol / wavelength_m / expected_freq_per_px
    assert abs(params.light_freq_px - expected_light_freq_px) < 1e-12

    # Test imgpx_m_per_px property
    expected_imgpx_m_per_px = px_size_m * img_size_px / expected_aperture
    assert abs(params.imgpx_m_per_px - expected_imgpx_m_per_px) < 1e-12

    # Test hologram2fourier property
    expected_hologram2fourier = (px_size_m / expected_freq_per_px) ** 0.5
    assert abs(params.hologram2spectrum - expected_hologram2fourier) < 1e-12

    # Test fourier2cpfield property
    expected_fourier2cpfield = (expected_freq_per_px / expected_imgpx_m_per_px) ** 0.5
    assert abs(params.spectrum2cpfield - expected_fourier2cpfield) < 1e-12

    # Test cpfield2spectrum property
    expected_cpfield2spectrum = (expected_imgpx_m_per_px / expected_freq_per_px) ** 0.5
    assert abs(params.cpfield2spectrum - expected_cpfield2spectrum) < 1e-12


def test_print_all_parameters(capsys: pytest.CaptureFixture[str]) -> None:
    params = MuParameters(na=1.4, wavelength_m=550e-9, img_size_px=100, px_size_m=6.5e-6, n_sol=1.33)

    # Test output when show_properties is False
    print_all_parameters(params, show_properties=False)
    captured = capsys.readouterr().out
    assert "=== Dataclass Parameters ===" in captured
    for field_obj in fields(params):
        assert f"{field_obj.name}:" in captured

    # Test output when show_properties is True
    print_all_parameters(params, show_properties=True)
    captured = capsys.readouterr().out
    assert "=== Properties ===" in captured


def test_make_disk() -> None:
    # Use numpy as backend
    center = (50, 50)
    radius = 10
    array_shape = (100, 100)

    # Lowpass mask: inside the circle should be True
    mask = make_disk(center, radius, array_shape, highpass=False)
    assert mask.dtype == jnp.bool_
    assert mask.shape == array_shape
    assert mask[50, 50]
    assert not mask[0, 0]

    # Highpass mask: outside the circle should be True
    mask_high = make_disk(center, radius, array_shape, highpass=True)
    assert mask_high.shape == array_shape
    assert not mask_high[50, 50]
    assert mask_high[0, 0]


def test_crop_array() -> None:
    # Create a 10x10 grid with sequential numbers
    arr = jnp.arange(100).reshape(10, 10)
    center = (5, 5)
    width = 5
    cropped = crop_array(arr, center, width)
    assert cropped.shape == (width, width)
    expected = arr[5 - width // 2 : 5 + width // 2 + 1, 5 - width // 2 : 5 + width // 2 + 1]
    assert jnp.array_equal(cropped, expected)


def test_get_spectrum() -> None:
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=64, px_size_m=1e-6, n_sol=1.33)
    offaxis_center = (32, 32)
    # Create a Fourier array with all ones
    ft_array = jnp.ones((params.img_size_px, params.img_size_px), dtype=float)

    spectrum = get_spectrum(ft_array.copy(), params, offaxis_center, crop_center=False)
    expected_shape = (params.aperturesize_px, params.aperturesize_px)
    assert spectrum.shape == expected_shape

    spectrum_crop = get_spectrum(ft_array.copy(), params, offaxis_center, crop_center=True, c_r=5)
    assert spectrum_crop.shape == expected_shape


def test_get_spectrums() -> None:
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=64, px_size_m=1e-6, n_sol=1.33)
    array = jnp.ones((params.img_size_px, params.img_size_px), dtype=float)
    offaxis_centers = [(32, 32), (20, 20)]
    spectrums = get_spectrums(array, params, offaxis_centers, crop_center=True, c_r=3)
    assert len(spectrums) == len(offaxis_centers)
    expected_shape = (params.aperturesize_px, params.aperturesize_px)
    for spec in spectrums:
        assert spec.shape == expected_shape


def test_correct_offset() -> None:
    # Create a small complex array
    array = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    offset_regs = [((0, 2), (0, 2))]
    phase_offset = jnp.mean(jnp.angle(array))
    amplitude_scale = jnp.mean(jnp.abs(array))
    expected = array * jnp.exp(-1j * phase_offset) / amplitude_scale

    corrected = correct_offset(array, offset_regs, phase=True, amplitude=True)
    assert jnp.allclose(corrected, expected, atol=1e-7)

    same_array = correct_offset(array, None, phase=True, amplitude=True)
    assert jnp.array_equal(same_array, array)


def test_correct_offset_flags() -> None:
    array = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
    offset_regs = [((0, 2), (0, 2))]

    # Test with phase correction disabled
    amplitude_scale = jnp.mean(jnp.abs(array))
    expected_phase_false = array / amplitude_scale
    corrected_phase_false = correct_offset(array, offset_regs, phase=False, amplitude=True)
    assert jnp.allclose(corrected_phase_false, expected_phase_false, atol=1e-7)

    # Test with amplitude correction disabled
    phase_offset = jnp.mean(jnp.angle(array))
    expected_amp_false = array * jnp.exp(-1j * phase_offset)
    corrected_amp_false = correct_offset(array, offset_regs, phase=True, amplitude=False)
    assert jnp.allclose(corrected_amp_false, expected_amp_false, atol=1e-7)


def test_meshgrid_freq() -> None:
    """Test meshgrid_freq function."""
    img_size_px = 8
    freq_per_px = 0.1
    fx, fy = meshgrid_freq(img_size_px, freq_per_px)

    assert fx.shape == (img_size_px, img_size_px)
    assert fy.shape == (img_size_px, img_size_px)

    # Check if frequencies match expected 1D frequency array
    expected_freq = jnp.fft.fftfreq(img_size_px, 1 / freq_per_px)
    expected_fx, expected_fy = jnp.meshgrid(expected_freq, expected_freq, indexing="ij")
    assert jnp.allclose(fx, expected_fx)
    assert jnp.allclose(fy, expected_fy)


def test_propagate_fresnel() -> None:
    """Test propagate_fresnel function with plane wave."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)

    # Create a plane wave (constant amplitude)
    plane_wave = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.complex64)
    z_obj_m = 1e-3  # 1mm propagation

    propagated = propagate_fresnel(plane_wave, params, z_obj_m)

    # For a plane wave, amplitude should remain approximately constant
    amplitude_variation = jnp.std(jnp.abs(propagated))
    assert amplitude_variation < 1e-6


def test_support_constraint() -> None:
    """Test support_constraint function."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)

    # Create a complex field
    field = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.complex64)

    constrained = support_constraint(field, params)

    # Check that values outside the support are zero
    mask = make_disk(params.img_center, radius=params.img_size_px // 4, array_shape=params.img_size_px)
    assert jnp.allclose(constrained, jnp.where(mask, field, 0))


def test_correct_offset_idh() -> None:
    """Test correct_offset_idh function."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)

    # Create a complex field with known phase and amplitude offset
    phase_offset = 0.5
    amp_scale = 2.0
    field = (
        amp_scale * jnp.exp(1j * phase_offset) * jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.complex64)
    )

    corrected = correct_offset_idh(field, params)

    # Check that the central region has been normalized
    roi = crop_array(corrected, params.img_center, width=16)
    assert jnp.allclose(jnp.mean(jnp.angle(roi)), 0, atol=1e-6)
    assert jnp.allclose(jnp.mean(jnp.abs(roi)), 1, atol=1e-6)


def test_inline_dh_basic() -> None:
    """Test basic functionality of inline_dh."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)

    # Create a simple hologram (intensity pattern)
    hologram = jnp.ones((params.img_size_px, params.img_size_px))
    z_obj_m = 1e-3  # 1mm object distance

    # Test without twin image suppression
    result = inline_dh(hologram, params, z_obj_m, twin_iter=0)

    assert result.shape == hologram.shape
    assert result.dtype == jnp.complex64


def test_inline_dh_parameter_validation() -> None:
    """Test parameter validation in inline_dh."""
    params = MuParameters(na=-0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)
    hologram = jnp.ones((32, 32))
    z_obj_m = 1e-3

    # Should raise ValueError due to negative NA
    with pytest.raises(ValueError, match="NA cannot be negative"):
        inline_dh(hologram, params, z_obj_m)


def test_inline_dh_negative_hologram() -> None:
    """Test that inline_dh raises error for negative hologram values."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)
    hologram = jnp.array([[-1.0, 1.0], [1.0, 1.0]])
    z_obj_m = 1e-3

    with pytest.raises(ValueError, match="Hologram cannot contain negative values"):
        inline_dh(hologram, params, z_obj_m)


def test_inline_dh_with_twin_suppression() -> None:
    """Test inline_dh with twin image suppression."""
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.0)
    hologram = jnp.ones((params.img_size_px, params.img_size_px))
    z_obj_m = 1e-3

    # Test with twin image suppression
    result = inline_dh(hologram, params, z_obj_m, twin_iter=2)

    assert result.shape == hologram.shape
    assert result.dtype == jnp.complex64


if __name__ == "__main__":
    pytest.main()
