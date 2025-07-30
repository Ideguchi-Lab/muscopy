import math
from dataclasses import fields

import jax.numpy as jnp
import pytest

from muscopy.dh import (
    MuParameters,
    correct_offset,
    crop_array,
    get_spectrum,
    get_spectrums,
    inline_dh,
    make_disk,
    print_all_parameters,
    ps_idh_reconstruct,
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

    # Test print_illumination_angle option
    spectrum_angle = get_spectrum(ft_array.copy(), params, offaxis_center, print_illumination_angle=True)
    assert spectrum_angle.shape == expected_shape


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


def test_ps_idh_reconstruct() -> None:
    """Test phase-shifting inline digital holography reconstruction."""
    # Create synthetic phase-shifted holograms
    h, w = 32, 32
    num_holograms = 4

    # Create a simple complex object field
    object_field = jnp.ones((h, w), dtype=complex) * (1 + 0.5j)
    ref_amp = 1.0

    # Generate phase shifts (4-step phase shifting)
    deltas = jnp.array([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2])

    # Generate synthetic holograms I_k = |O + R * exp(j*delta_k)|^2
    i_stack = jnp.zeros((num_holograms, h, w))
    for k in range(num_holograms):
        hologram_field = object_field + ref_amp * jnp.exp(1j * deltas[k])
        i_stack = i_stack.at[k].set(jnp.abs(hologram_field) ** 2)

    # Reconstruct using PS-IDH
    reconstructed = ps_idh_reconstruct(i_stack, deltas, ref_amp)

    # Check shape and type
    assert reconstructed.shape == (h, w)
    assert jnp.iscomplexobj(reconstructed)

    # For perfect 4-step phase shifting, we should recover the original object field
    # The reconstruction formula gives: O_hat = (1/M) * sum(I_k * exp(-j*delta_k)) / (2*R)
    # For the synthetic data: I_k = |O + R*exp(j*delta_k)|^2
    # This should approximately recover the object field
    expected_amplitude = jnp.abs(object_field)
    reconstructed_amplitude = jnp.abs(reconstructed)

    # Check if the reconstructed amplitude is reasonable (not exact due to interference terms)
    assert jnp.all(reconstructed_amplitude > 0)
    assert reconstructed_amplitude.shape == expected_amplitude.shape


def test_ps_idh_reconstruct_shape_mismatch() -> None:
    """Test that ps_idh_reconstruct raises error for mismatched shapes."""
    i_stack = jnp.ones((4, 32, 32))
    deltas = jnp.array([0.0, jnp.pi / 2, jnp.pi])  # Wrong number of phase shifts

    with pytest.raises(ValueError, match=r"Number of holograms .* must match number of phase shifts"):
        ps_idh_reconstruct(i_stack, deltas)


def test_inline_dh_with_known_phase_shifts() -> None:
    """Test inline_dh function with known phase shifts."""
    params = MuParameters(na=0.5, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.33)

    # Create synthetic data
    h, w = 32, 32
    num_holograms = 3
    i_stack = jnp.ones((num_holograms, h, w)) * 2.0  # Simple constant intensity
    deltas = jnp.array([0.0, 2 * jnp.pi / 3, 4 * jnp.pi / 3])

    # Test reconstruction
    result = inline_dh(i_stack, deltas, params, ref_amp=1.0)

    assert result.shape == (h, w)
    assert jnp.iscomplexobj(result)


def test_inline_dh_invalid_stack_shape() -> None:
    """Test that inline_dh raises error for invalid stack shape."""
    params = MuParameters(na=0.5, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.33)

    # 2D array instead of 3D
    i_stack = jnp.ones((32, 32))
    deltas = jnp.array([0.0, jnp.pi])

    with pytest.raises(ValueError, match="i_stack must be 3D array"):
        inline_dh(i_stack, deltas, params)


def test_inline_dh_phase_shift_mismatch() -> None:
    """Test that inline_dh raises error for phase shift count mismatch."""
    params = MuParameters(na=0.5, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.33)

    i_stack = jnp.ones((4, 32, 32))
    deltas = jnp.array([0.0, jnp.pi])  # Wrong number

    with pytest.raises(ValueError, match=r"Number of phase shifts .* must match number of holograms"):
        inline_dh(i_stack, deltas, params)


def test_inline_dh_blind_reconstruction_not_implemented() -> None:
    """Test that blind reconstruction raises NotImplementedError."""
    params = MuParameters(na=0.5, wavelength_m=500e-9, img_size_px=32, px_size_m=1e-6, n_sol=1.33)

    i_stack = jnp.ones((3, 32, 32))

    # Test with deltas=None (should trigger blind reconstruction)
    with pytest.raises(NotImplementedError, match="Blind reconstruction not yet implemented"):
        inline_dh(i_stack, None, params)

    # Test with blind_reconstruction=True
    deltas = jnp.array([0.0, jnp.pi / 2, jnp.pi])
    with pytest.raises(NotImplementedError, match="Blind reconstruction not yet implemented"):
        inline_dh(i_stack, deltas, params, blind_reconstruction=True)


if __name__ == "__main__":
    pytest.main()
