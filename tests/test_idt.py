"""Test cases for IDT module."""

from collections.abc import Sequence

import jax.numpy as jnp
import pytest
from jax import Array

from muscopy import idt as idt_module
from muscopy.cfg import ArrayPrecision
from muscopy.idt import (
    IDTConfig,
    IDTParameters,
    IDTZCenteringMode,
    compute_g_list,
    compute_idt,
    compute_permitivity,
    compute_permittivity,
    fourier_transform,
    idt_coherent_pupil_radius_px,
    idt_intensity_support_radius_px,
    make_frequency_grid_xy,
    make_green_func,
    make_pupil_func,
    make_z_position,
    transfer_func_im,
    transfer_func_re,
    validate_idt_params,
)


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


def test_make_frequency_grid_xy_uses_axis_0_for_kx_and_axis_1_for_ky() -> None:
    params = _small_idt_params()
    precision = ArrayPrecision()
    coords = jnp.arange(-params.aperturesize_px, params.aperturesize_px + 1, dtype=jnp.float32)

    kx, ky = make_frequency_grid_xy(params, precision=precision)

    expected_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    assert kx.shape == expected_shape
    assert ky.shape == expected_shape
    assert kx.dtype == jnp.float32
    assert ky.dtype == jnp.float32
    assert bool(jnp.allclose(kx[:, 0], coords))
    assert bool(jnp.allclose(kx[0, :], -params.aperturesize_px))
    assert bool(jnp.allclose(ky[0, :], coords))
    assert bool(jnp.allclose(ky[:, 0], -params.aperturesize_px))


def test_idt_aperture_radii_document_intensity_and_coherent_support() -> None:
    params = _small_idt_params()

    assert idt_intensity_support_radius_px(params) == params.aperturesize_px
    assert bool(jnp.allclose(idt_coherent_pupil_radius_px(params), params.aperturesize_px / 2))
    assert params.intensity_support_radius_px == params.aperturesize_px
    assert bool(jnp.allclose(params.coherent_pupil_radius_px, params.aperturesize_px / 2))


def test_fourier_transform_masks_to_intensity_support_radius() -> None:
    params = _small_idt_params()
    precision = ArrayPrecision()
    image_xy = jnp.arange(params.img_size_px**2, dtype=jnp.float32).reshape(params.img_size_px, params.img_size_px)
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    expected_support = kx**2 + ky**2 <= params.intensity_support_radius_px**2

    [spectrum] = fourier_transform(params, [image_xy], precision=precision)

    assert spectrum.shape == (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    assert bool(jnp.allclose(spectrum[~expected_support], 0.0))


def test_make_pupil_func_uses_half_radius_coherent_pupil_support() -> None:
    params = _small_idt_params()
    precision = ArrayPrecision()
    kx, ky = make_frequency_grid_xy(params, precision=precision)

    pupil = make_pupil_func(params, (0.0, 0.0), precision=precision)

    expected = kx**2 + ky**2 <= params.coherent_pupil_radius_px**2
    assert bool(jnp.all(pupil == expected))


def test_make_pupil_func_shifts_subpixel_support_on_requested_axis() -> None:
    params = IDTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=64,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
        num_z_slices=2,
    )
    precision = ArrayPrecision()
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    du = 0.5

    pupil_x = make_pupil_func(params, (du, 0.0), precision=precision)
    pupil_y = make_pupil_func(params, (0.0, du), precision=precision)

    center_x_for_x_shift = jnp.sum(kx * pupil_x) / jnp.sum(pupil_x)
    center_y_for_x_shift = jnp.sum(ky * pupil_x) / jnp.sum(pupil_x)
    center_x_for_y_shift = jnp.sum(kx * pupil_y) / jnp.sum(pupil_y)
    center_y_for_y_shift = jnp.sum(ky * pupil_y) / jnp.sum(pupil_y)
    assert bool(jnp.allclose(center_x_for_x_shift, -du, atol=0.05))
    assert bool(jnp.allclose(center_y_for_x_shift, 0.0, atol=0.05))
    assert bool(jnp.allclose(center_x_for_y_shift, 0.0, atol=0.05))
    assert bool(jnp.allclose(center_y_for_y_shift, -du, atol=0.05))


def test_make_green_func_zeros_values_outside_shared_support() -> None:
    params = _small_idt_params()
    precision = ArrayPrecision()
    u_shift = (0.5, 0.0)
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    ux = kx + u_shift[0]
    uy = ky + u_shift[1]
    expected_support = (ux**2 + uy**2 <= (params.aperturesize_px / 2) ** 2) & (
        params.light_freq_px**2 - ux**2 - uy**2 > 0
    )

    green = make_green_func(params, u_shift, z=0.0, precision=precision)

    assert green.dtype == jnp.complex64
    assert bool(jnp.allclose(green[~expected_support], 0.0))
    assert bool(jnp.all(jnp.isfinite(green[expected_support])))
    assert bool(jnp.any(jnp.abs(green[expected_support]) > 0.0))


def test_make_z_position_defaults_to_central_slice_zero_convention() -> None:
    params = IDTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=32,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
        num_z_slices=10,
    )
    config = IDTConfig()

    z_positions = jnp.array([make_z_position(params, idx_z, config) for idx_z in range(params.num_z_slices)])

    expected = jnp.arange(-5, 5, dtype=jnp.float32) * params.imgpx_axial_m_per_px
    assert config.z_centering == IDTZCenteringMode.CENTRAL_SLICE_ZERO
    assert bool(jnp.allclose(z_positions, expected))


def test_make_z_position_supports_symmetric_volume_convention() -> None:
    params = IDTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=32,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
        num_z_slices=10,
    )
    config = IDTConfig(z_centering="symmetric_volume")

    z_positions = jnp.array([make_z_position(params, idx_z, config) for idx_z in range(params.num_z_slices)])

    expected = (jnp.arange(10, dtype=jnp.float32) - 4.5) * params.imgpx_axial_m_per_px
    assert config.z_centering == IDTZCenteringMode.SYMMETRIC_VOLUME
    assert bool(jnp.allclose(z_positions, expected))


def test_compute_g_list_normalizes_by_per_pixel_reference() -> None:
    target = jnp.array([[2.0, 8.0], [3.0, 4.0]], dtype=jnp.float32)
    reference = jnp.array([[1.0, 4.0], [6.0, 2.0]], dtype=jnp.float32)

    [g] = compute_g_list([target], [reference], normalize=True)

    expected = jnp.array([[1.0, 1.0], [-0.5, 1.0]], dtype=jnp.float32)
    assert g.dtype == jnp.float32
    assert bool(jnp.allclose(g, expected))


def test_compute_g_list_masks_zero_reference_pixels() -> None:
    target = jnp.array([[1.0, 4.0]], dtype=jnp.float32)
    reference = jnp.array([[0.0, 2.0]], dtype=jnp.float32)

    [g] = compute_g_list([target], [reference], normalize=True)

    expected = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
    assert g.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(g)))
    assert bool(jnp.allclose(g, expected))


def test_compute_g_list_clips_normalized_reference_outliers() -> None:
    target = jnp.array([[11.0]], dtype=jnp.float32)
    reference = jnp.array([[1.0]], dtype=jnp.float32)

    [g] = compute_g_list([target], [reference], normalize=True, g_clip=5.0)

    assert bool(jnp.allclose(g, 5.0))


def test_compute_g_list_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="i_list and i_reference must have the same length"):
        compute_g_list([jnp.ones((1, 1)), jnp.ones((1, 1))], [jnp.ones((1, 1))])


def test_compute_g_list_rejects_invalid_ref_floor_ratio() -> None:
    with pytest.raises(ValueError, match="ref_floor_ratio must be non-negative"):
        compute_g_list([jnp.ones((1, 1))], [jnp.ones((1, 1))], ref_floor_ratio=-0.1)


def test_compute_g_list_rejects_invalid_g_clip() -> None:
    with pytest.raises(ValueError, match="g_clip must be positive"):
        compute_g_list([jnp.ones((1, 1))], [jnp.ones((1, 1))], g_clip=0.0)


def test_compute_idt_rejects_non_xy_image_shape() -> None:
    params = _small_idt_params()
    bad_image = jnp.ones((params.img_size_px, params.img_size_px, 1), dtype=jnp.float32)
    reference = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"intensity_images\[0\] must be a 2D \[x, y\] array"):
        compute_idt(params, [bad_image], [reference], [(0.0, 0.0)])


def test_compute_idt_rejects_wrong_xy_image_size() -> None:
    params = _small_idt_params()
    bad_image = jnp.ones((params.img_size_px - 1, params.img_size_px), dtype=jnp.float32)
    reference = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)

    with pytest.raises(ValueError, match=r"shape \(8, 8\) in \[x, y\] order"):
        compute_idt(params, [bad_image], [reference], [(0.0, 0.0)])


def test_compute_permitivity_restores_normalized_transfer_scale(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    g_tilde = jnp.full(spectrum_shape, 6.0 + 0.0j, dtype=jnp.complex64)

    def fake_transfer_func_re(
        params_arg: IDTParameters,
        u_illumination: tuple[float, float],
        z: float,
        incident_intensity: float,
        *,
        precision: ArrayPrecision | None = None,
    ) -> Array:
        del params_arg, u_illumination, z, incident_intensity, precision
        return jnp.full(spectrum_shape, 2.0 + 0.0j, dtype=jnp.complex64)

    def fake_transfer_func_im(
        params_arg: IDTParameters,
        u_illumination: tuple[float, float],
        z: float,
        incident_intensity: float,
        *,
        precision: ArrayPrecision | None = None,
    ) -> Array:
        del params_arg, u_illumination, z, incident_intensity, precision
        return jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    monkeypatch.setattr(idt_module, "transfer_func_re", fake_transfer_func_re)
    monkeypatch.setattr(idt_module, "transfer_func_im", fake_transfer_func_im)

    eps_re, eps_im = compute_permitivity(
        params,
        [g_tilde],
        [(0.0, 0.0)],
        [1.0],
        alpha=0.0,
        beta=1.0,
    )

    assert bool(jnp.allclose(eps_re, 3.0 + 0.0j))
    assert bool(jnp.allclose(eps_im, 0.0 + 0.0j))


def test_compute_permittivity_keeps_backward_compatible_typo_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    g_tilde = jnp.full(spectrum_shape, 2.0 + 0.0j, dtype=jnp.complex64)

    def fake_transfer_func_re(
        params_arg: IDTParameters,
        u_illumination: tuple[float, float],
        z: float,
        incident_intensity: float,
        *,
        precision: ArrayPrecision | None = None,
    ) -> Array:
        del params_arg, u_illumination, z, incident_intensity, precision
        return jnp.ones(spectrum_shape, dtype=jnp.complex64)

    def fake_transfer_func_im(
        params_arg: IDTParameters,
        u_illumination: tuple[float, float],
        z: float,
        incident_intensity: float,
        *,
        precision: ArrayPrecision | None = None,
    ) -> Array:
        del params_arg, u_illumination, z, incident_intensity, precision
        return jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    monkeypatch.setattr(idt_module, "transfer_func_re", fake_transfer_func_re)
    monkeypatch.setattr(idt_module, "transfer_func_im", fake_transfer_func_im)

    fixed_name_result = compute_permittivity(params, [g_tilde], [(0.0, 0.0)], [1.0], alpha=0.0, beta=1.0)
    typo_alias_result = compute_permitivity(params, [g_tilde], [(0.0, 0.0)], [1.0], alpha=0.0, beta=1.0)

    assert bool(jnp.allclose(fixed_name_result[0], typo_alias_result[0]))
    assert bool(jnp.allclose(fixed_name_result[1], typo_alias_result[1]))


def test_compute_permitivity_zeros_singular_inverse_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    g_tilde = jnp.full(spectrum_shape, 6.0 + 0.0j, dtype=jnp.complex64)

    def fake_transfer_func(
        params_arg: IDTParameters,
        u_illumination: tuple[float, float],
        z: float,
        incident_intensity: float,
        *,
        precision: ArrayPrecision | None = None,
    ) -> Array:
        del params_arg, u_illumination, z, incident_intensity, precision
        return jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    monkeypatch.setattr(idt_module, "transfer_func_re", fake_transfer_func)
    monkeypatch.setattr(idt_module, "transfer_func_im", fake_transfer_func)

    eps_re, eps_im = compute_permitivity(
        params,
        [g_tilde],
        [(0.0, 0.0)],
        [1.0],
        alpha=0.0,
        beta=0.0,
    )

    assert bool(jnp.all(jnp.isfinite(eps_re)))
    assert bool(jnp.all(jnp.isfinite(eps_im)))
    assert bool(jnp.allclose(eps_re, 0.0 + 0.0j))
    assert bool(jnp.allclose(eps_im, 0.0 + 0.0j))


def test_transfer_func_re_allows_tiny_negative_illumination_roundoff() -> None:
    params = _small_idt_params()
    light_freq_px = params.light_freq_px
    u_illumination = (float((light_freq_px**2 + 0.5e-9) ** 0.5), 0.0)

    transfer_func = transfer_func_re(params, u_illumination, z=0.0, incident_intensity=1.0)

    assert transfer_func.shape == (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    assert bool(jnp.all(jnp.isfinite(transfer_func)))


def test_transfer_func_re_rejects_evanescent_illumination_beyond_tolerance() -> None:
    params = _small_idt_params()
    light_freq_px = params.light_freq_px
    u_illumination = (float((light_freq_px**2 + 2e-9) ** 0.5), 0.0)

    with pytest.raises(ValueError, match="u_ill_z_squared must be non-negative"):
        transfer_func_re(params, u_illumination, z=0.0, incident_intensity=1.0)


def test_transfer_funcs_have_no_fast_axial_carrier_on_axis_dc() -> None:
    params = _small_idt_params()
    precision = ArrayPrecision()
    center = params.aperturesize_px
    u_illumination = (0.0, 0.0)
    z_values = jnp.arange(-2, 3, dtype=jnp.float32) * params.imgpx_axial_m_per_px

    h_re_center = []
    h_im_center = []
    for z in z_values:
        z_float = float(z)
        h_re = transfer_func_re(
            params,
            u_illumination,
            z_float,
            incident_intensity=1.0,
            precision=precision,
        )
        h_im = transfer_func_im(
            params,
            u_illumination,
            z_float,
            incident_intensity=1.0,
            precision=precision,
        )
        h_re_center.append(h_re[center, center])
        h_im_center.append(h_im[center, center])

    h_re_center_array = jnp.asarray(h_re_center)
    h_im_center_array = jnp.asarray(h_im_center)

    assert bool(jnp.max(jnp.abs(h_re_center_array)) < 1e-5)
    assert bool(
        jnp.std(jnp.real(h_im_center_array)) < 1e-5 * jnp.maximum(1.0, jnp.abs(jnp.mean(jnp.real(h_im_center_array)))),
    )


def test_compute_permitivity_rejects_invalid_led_intensity() -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    g_tilde = jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    with pytest.raises(ValueError, match=r"led_illumination_intensities\[0\] must be positive and finite"):
        compute_permitivity(
            params,
            [g_tilde],
            [(0.0, 0.0)],
            [0.0],
        )


def test_compute_permitivity_rejects_invalid_determinant_rel_floor() -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    g_tilde = jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    with pytest.raises(ValueError, match="determinant_rel_floor must be non-negative"):
        compute_permitivity(
            params,
            [g_tilde],
            [(0.0, 0.0)],
            [1.0],
            determinant_rel_floor=-0.1,
        )


def test_validate_idt_params_rejects_intensity_support_larger_than_image() -> None:
    params = IDTParameters(
        na=1.0,
        wavelength_m=1.0,
        img_size_px=4,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=1.0,
        num_z_slices=2,
    )

    with pytest.raises(ValueError, match=r"2 \* aperturesize_px \+ 1 must be <= img_size_px"):
        validate_idt_params(params)


def test_compute_idt_docstring_describes_coupled_inverse() -> None:
    assert compute_idt.__doc__ is not None
    assert "off-diagonal coupling term" in compute_idt.__doc__
    assert "estimated jointly" in compute_idt.__doc__


def test_compute_idt_can_warn_on_ifft_imaginary_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    params = _small_idt_params()
    spectrum_shape = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1)
    intensity_images = [jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)]
    ref_intensity_images = [jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.float32)]

    def fake_compute_permittivity(
        params_arg: IDTParameters,
        g_tilde_list: Sequence[Array],
        u_illumination_list: Sequence[tuple[float, float]],
        led_illumination_intensities: Sequence[float],
        z: float = 0.0,
        alpha: float = 1e-6,
        beta: float = 1e-6,
        *,
        precision: ArrayPrecision | None = None,
        determinant_rel_floor: float = 1e-4,
    ) -> tuple[Array, Array]:
        del (
            params_arg,
            g_tilde_list,
            u_illumination_list,
            led_illumination_intensities,
            z,
            alpha,
            beta,
            precision,
            determinant_rel_floor,
        )
        return jnp.ones(spectrum_shape, dtype=jnp.complex64) * 1j, jnp.zeros(spectrum_shape, dtype=jnp.complex64)

    monkeypatch.setattr(idt_module, "compute_permittivity", fake_compute_permittivity)
    config = IDTConfig(check_ifft_imag_residual=True, imag_residual_warn_threshold=1e-6)

    with pytest.warns(RuntimeWarning, match="eps_re inverse FFT imaginary residual"):
        compute_idt(params, intensity_images, ref_intensity_images, [(0.0, 0.0)], config=config)


def test_compute_idt_rejects_mutated_64_bit_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("muscopy.cfg._jax_x64_enabled", lambda: False)
    params = _small_idt_params()
    precision = ArrayPrecision()
    precision.float_length = 64
    config = IDTConfig(precision=precision)

    with pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"):
        compute_idt(params, [], [], [], config=config)
