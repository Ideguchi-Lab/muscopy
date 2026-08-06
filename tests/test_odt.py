"""Test cases for ODT module."""

import math
import warnings
from collections.abc import Sequence

import jax.numpy as jnp
import pytest
from jax import Array

from muscopy import odt as odt_module
from muscopy.cfg import ArrayPrecision
from muscopy.odt import (
    EwaldEmbeddingMode,
    ODTConfig,
    ODTParameters,
    ScatteringSpectrum,
    calc_scattering_potential_from_spectrums,
    calc_scattering_spectrums,
    calculate_odt_difference,
    synthesize_spectrum,
)


def test_odt_config_defaults_to_32_bit_precision() -> None:
    """Test that ODT defaults to the portable precision contract."""
    config = ODTConfig()

    assert config.precision.int_precision() == "int32"
    assert config.precision.float_precision() == "float32"
    assert config.precision.complex_precision() == "complex64"
    assert config.gradient_correction is True
    assert config.ewald_embedding_mode is EwaldEmbeddingMode.TRUNCATE
    assert config.verbose is False


def test_synthesize_spectrum_is_silent_by_default(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that ODT library calls do not print progress by default."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )

    synthesize_spectrum([], params, ODTConfig())

    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


def test_synthesize_spectrum_prints_when_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    """Test that ODT progress output remains opt-in."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(verbose=True)

    synthesize_spectrum([], params, config)

    assert "Synthesize spectrum..." in capsys.readouterr().out


def test_calc_scattering_spectrums_rejects_mutated_64_bit_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ODT validates precision before reconstruction."""
    monkeypatch.setattr("muscopy.cfg._jax_x64_enabled", lambda: False)
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    precision = ArrayPrecision()
    precision.float_length = 64
    config = ODTConfig(precision=precision)

    with pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"):
        calc_scattering_spectrums([], [], params, config)


def test_synthesize_spectrum_applies_scattering_spectrum_coefficient() -> None:
    """Test that weighted ODT synthesis applies per-spectrum linear coefficients."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    spectrum_shape = 2 * params.aperturesize_px + 1
    spectrum = jnp.ones((spectrum_shape, spectrum_shape), dtype=jnp.complex64)
    coefficient = 2.0 + 0.5j

    unweighted = synthesize_spectrum([ScatteringSpectrum(spectrum, (0, 0))], params, config)
    weighted = synthesize_spectrum([ScatteringSpectrum(spectrum, (0, 0), coefficient)], params, config)

    assert jnp.allclose(weighted, coefficient * unweighted)


def test_synthesize_spectrum_keeps_coverage_count_independent_from_coefficients() -> None:
    """Test that weighted synthesis still averages overlaps by support count."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    spectrum_shape = 2 * params.aperturesize_px + 1
    spectrum = jnp.ones((spectrum_shape, spectrum_shape), dtype=jnp.complex64)
    coefficient_1 = 2.0 + 0.0j
    coefficient_2 = 6.0 + 0.0j

    single = synthesize_spectrum([ScatteringSpectrum(spectrum, (0, 0))], params, config)
    overlapped = synthesize_spectrum(
        [
            ScatteringSpectrum(spectrum, (0, 0), coefficient_1),
            ScatteringSpectrum(spectrum, (0, 0), coefficient_2),
        ],
        params,
        config,
    )

    assert jnp.allclose(overlapped, ((coefficient_1 + coefficient_2) / 2) * single)


def test_synthesize_spectrum_counts_zero_values_as_observed_support() -> None:
    """Test that zero-valued measurements still contribute to ODT overlap averaging."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    spectrum_shape = 2 * params.aperturesize_px + 1
    measured_spectrum = jnp.ones((spectrum_shape, spectrum_shape), dtype=jnp.complex64)
    zero_spectrum = jnp.zeros_like(measured_spectrum)

    single = synthesize_spectrum([ScatteringSpectrum(measured_spectrum, (0, 0))], params, config)
    overlapped_with_zero = synthesize_spectrum(
        [
            ScatteringSpectrum(measured_spectrum, (0, 0)),
            ScatteringSpectrum(zero_spectrum, (0, 0)),
        ],
        params,
        config,
    )
    observed_mask = jnp.abs(single) > 0

    assert jnp.any(observed_mask)
    assert jnp.allclose(overlapped_with_zero[observed_mask], single[observed_mask] / 2)


def test_shift_dh_spectrum_preserves_center_for_zero_illumination() -> None:
    """Test that zero illumination keeps a centered DH spectrum at the expanded center."""
    params = ODTParameters(
        na=1.2,
        wavelength_m=532e-9,
        img_size_px=64,
        px_size_m=100e-9,
        n_sol=1.33,
        na_illumination=0.9,
    )
    cp_spectrum = jnp.zeros((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)
    cp_spectrum = cp_spectrum.at[params.aperturesize_px // 2, params.aperturesize_px // 2].set(1)

    shifted = odt_module._shift_dh_spectrum(params, cp_spectrum, (0, 0))  # noqa: SLF001
    shifted_with_edge = odt_module._shift_dh_spectrum(params, cp_spectrum, (0, 0), edge_size=2)  # noqa: SLF001

    assert shifted.shape == (59, 59)
    assert tuple(index.item() for index in jnp.unravel_index(jnp.argmax(jnp.abs(shifted)), shifted.shape)) == (29, 29)
    assert tuple(
        index.item() for index in jnp.unravel_index(jnp.argmax(jnp.abs(shifted_with_edge)), shifted_with_edge.shape)
    ) == (31, 31)


def test_calc_ewald_embedding_weight_supports_explicit_placement_modes() -> None:
    """Test Ewald truncation, nearest-plane, and linear axial interpolation semantics."""
    params = ODTParameters(
        na=0.5,
        wavelength_m=1.0,
        img_size_px=10,
        px_size_m=1.0,
        n_sol=1.0,
        na_illumination=0.5,
    )
    illumination_vector = (4, 0)
    shape_3d = (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1, params.freq_axial_extent_px)
    coord_x, coord_y = -5, 0
    x_index = coord_x - (-shape_3d[0] // 2 + 1)
    y_index = coord_y - (-shape_3d[1] // 2 + 1)
    z_zero_index = 0 - (-shape_3d[2] // 2 + 1)
    z_one_index = 1 - (-shape_3d[2] // 2 + 1)
    fz_value = math.sqrt(params.light_freq_px**2 - (coord_x + illumination_vector[0]) ** 2) - math.sqrt(
        params.light_freq_px**2 - illumination_vector[0] ** 2
    )

    truncated = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
        shape_3d,
        params,
        illumination_vector,
        "Forward",
        EwaldEmbeddingMode.TRUNCATE,
    )
    nearest = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
        shape_3d,
        params,
        illumination_vector,
        "Forward",
        EwaldEmbeddingMode.NEAREST,
    )
    linear = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
        shape_3d,
        params,
        illumination_vector,
        "Forward",
        EwaldEmbeddingMode.LINEAR,
    )

    assert 0.5 < fz_value < 1
    assert truncated[x_index, y_index, z_zero_index] == 1
    assert nearest[x_index, y_index, z_one_index] == 1
    assert jnp.allclose(linear[x_index, y_index, z_zero_index], 1 - fz_value)
    assert jnp.allclose(linear[x_index, y_index, z_one_index], fz_value)
    assert jnp.allclose(jnp.sum(linear[x_index, y_index, :]), 1)


def test_synthesize_spectrum_uses_configured_ewald_embedding_mode() -> None:
    """Test that ODT synthesis applies the Ewald embedding mode from ODTConfig."""
    params = ODTParameters(
        na=0.5,
        wavelength_m=1.0,
        img_size_px=10,
        px_size_m=1.0,
        n_sol=1.0,
        na_illumination=0.5,
    )
    illumination_vector = (4, 0)
    spectrum_shape = 2 * params.aperturesize_px + 1
    coord_x, coord_y = -5, 0
    x_index = coord_x - (-spectrum_shape // 2 + 1)
    y_index = coord_y - (-spectrum_shape // 2 + 1)
    z_zero_index = 0 - (-params.freq_axial_extent_px // 2 + 1)
    z_one_index = 1 - (-params.freq_axial_extent_px // 2 + 1)
    spectrum2d = jnp.zeros((spectrum_shape, spectrum_shape), dtype=jnp.complex64)
    spectrum2d = spectrum2d.at[x_index, y_index].set(1)

    truncated = synthesize_spectrum(
        [ScatteringSpectrum(spectrum2d, illumination_vector)],
        params,
        ODTConfig(hermite_symmetry=False, ewald_embedding_mode=EwaldEmbeddingMode.TRUNCATE),
    )
    linear = synthesize_spectrum(
        [ScatteringSpectrum(spectrum2d, illumination_vector)],
        params,
        ODTConfig(hermite_symmetry=False, ewald_embedding_mode=EwaldEmbeddingMode.LINEAR),
    )

    assert truncated[x_index, y_index, z_zero_index] != 0
    assert truncated[x_index, y_index, z_one_index] == 0
    assert linear[x_index, y_index, z_zero_index] != 0
    assert jnp.allclose(linear[x_index, y_index, z_one_index], linear[x_index, y_index, z_zero_index])


def test_linear_ewald_embedding_is_finite_outside_propagating_circle() -> None:
    """Test that masked evanescent coordinates do not contaminate linear weights with NaNs."""
    params = ODTParameters(
        na=1.1,
        wavelength_m=532e-9,
        img_size_px=64,
        px_size_m=3.45e-6 * 3 / 180 / 2,
        n_sol=1.33,
        na_illumination=1.0,
    )
    shape_3d = (
        2 * params.aperturesize_px + 1,
        2 * params.aperturesize_px + 1,
        params.freq_axial_extent_px,
    )

    weights = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
        shape_3d,
        params,
        (3, 0),
        "Forward",
        EwaldEmbeddingMode.LINEAR,
    )

    assert bool(jnp.all(jnp.isfinite(weights)))


def test_odt_config_rejects_unknown_ewald_embedding_mode() -> None:
    """Test that invalid Ewald embedding modes fail explicitly at configuration time."""
    with pytest.raises(TypeError, match="ewald_embedding_mode must be an EwaldEmbeddingMode"):
        ODTConfig(
            hermite_symmetry=False,
            ewald_embedding_mode="invalid",  # type: ignore[arg-type]  # Runtime validation test.
        )


def test_calc_1st_scattering_spectrum_respects_gradient_correction_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that ODT gradient correction remains configurable."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    cp_field = jnp.ones((7, 7), dtype=jnp.complex64) * 2
    ref_cp_field = jnp.ones_like(cp_field)
    calls = []

    def fake_correct_gradient(array: Array, *, edge_size: int = 0) -> Array:
        calls.append(edge_size)
        return array

    monkeypatch.setattr(odt_module, "correct_gradient", fake_correct_gradient)

    odt_module._calc_1st_scattering_spectrum(  # noqa: SLF001
        cp_field,
        ref_cp_field,
        params,
        "Born",
        (0, 0),
        gradient_correction=False,
    )
    assert calls == []

    odt_module._calc_1st_scattering_spectrum(cp_field, ref_cp_field, params, "Born", (0, 0))  # noqa: SLF001
    assert calls == [0]


def test_calc_scattering_spectrums_uses_configured_gradient_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that public ODT spectrum extraction applies gradient correction config."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    spectrum_shape = params.aperturesize_px
    cp_spectrum = jnp.ones((spectrum_shape, spectrum_shape), dtype=jnp.complex64) * 2
    ref_cp_spectrum = jnp.ones_like(cp_spectrum)
    calls = []

    def fake_correct_gradient(array: Array, *, edge_size: int = 0) -> Array:
        calls.append(edge_size)
        return array

    monkeypatch.setattr(odt_module, "correct_gradient", fake_correct_gradient)

    calc_scattering_spectrums(
        [cp_spectrum],
        [ref_cp_spectrum],
        params,
        ODTConfig(gradient_correction=False),
        illumination_vectors=[(0, 0)],
    )
    assert calls == []

    calc_scattering_spectrums(
        [cp_spectrum],
        [ref_cp_spectrum],
        params,
        ODTConfig(),
        illumination_vectors=[(0, 0)],
    )
    assert calls == [0]


def test_log_field_preserves_low_amplitude_complex_phase() -> None:
    """Test that Rytov log stabilization does not perturb low-amplitude phase."""
    cp_field = jnp.full((3, 3), 1j * odt_module.EPSILON / 10, dtype=jnp.complex64)
    ref_cp_field = jnp.ones_like(cp_field)

    log_field = odt_module._log_field(cp_field, ref_cp_field)  # noqa: SLF001

    assert jnp.allclose(jnp.imag(log_field), jnp.pi / 2, atol=1e-5)


def test_calc_scattering_potential_from_spectrums_matches_factored_reconstruction_path() -> None:
    """Test the explicit factored ODT path."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    cp_spectrum = jnp.ones((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)
    ref_cp_spectrum = jnp.ones_like(cp_spectrum)

    scattering_spectrums = calc_scattering_spectrums([cp_spectrum], [ref_cp_spectrum], params, config)
    factored_potential, factored_spectrum = calc_scattering_potential_from_spectrums(
        scattering_spectrums,
        params,
        config,
    )
    repeated_scattering_spectrums = calc_scattering_spectrums([cp_spectrum], [ref_cp_spectrum], params, config)
    repeated_potential, repeated_spectrum = calc_scattering_potential_from_spectrums(
        repeated_scattering_spectrums,
        params,
        config,
    )

    assert jnp.allclose(repeated_potential, factored_potential)
    assert jnp.allclose(repeated_spectrum, factored_spectrum)


def test_odt_remains_supported_without_deprecation_warning() -> None:
    """Test that the high-level ODT wrapper remains supported."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    cp_spectrum = jnp.ones((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)
    ref_cp_spectrum = jnp.ones_like(cp_spectrum)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        refractive_index, synthesized_spectrum = odt_module.odt([cp_spectrum], [ref_cp_spectrum], params, config)

    assert not [warning for warning in caught_warnings if issubclass(warning.category, FutureWarning)]
    assert refractive_index.ndim == 3
    assert synthesized_spectrum.ndim == 3


def test_calc_scattering_spectrums_preserves_explicit_illumination_vectors() -> None:
    """Test that explicit illumination vectors bypass reference peak inference."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    cp_spectrum = jnp.ones((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)
    ref_cp_spectrum = jnp.ones_like(cp_spectrum)

    scattering_spectrums = calc_scattering_spectrums(
        [cp_spectrum],
        [ref_cp_spectrum],
        params,
        config,
        illumination_vectors=[(0, 0)],
    )

    assert len(scattering_spectrums) == 1
    assert scattering_spectrums[0].illumination_vector == (0, 0)


def test_calc_scattering_spectrums_rejects_illumination_vector_count_mismatch() -> None:
    """Test that explicit illumination metadata must align with the input spectra."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    cp_spectrum = jnp.ones((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)
    ref_cp_spectrum = jnp.ones_like(cp_spectrum)

    with pytest.raises(ValueError, match="illumination_vectors must match"):
        calc_scattering_spectrums(
            [cp_spectrum],
            [ref_cp_spectrum],
            params,
            config,
            illumination_vectors=[],
        )


def test_calc_scattering_spectrums_rejects_reference_count_mismatch() -> None:
    """Test that target and reference spectrum counts must match."""
    params = ODTParameters(
        na=0.1,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.1,
    )
    config = ODTConfig(hermite_symmetry=False)
    cp_spectrum = jnp.ones((params.aperturesize_px, params.aperturesize_px), dtype=jnp.complex64)

    with pytest.raises(ValueError, match="cp_spectrums and ref_cp_spectrums must have the same length"):
        calc_scattering_spectrums([cp_spectrum], [], params, config)


class TestCalculateODTDifference:
    """Test cases for calculate_odt_difference function."""

    def setup_method(self) -> None:
        """Set up test parameters."""
        self.params = ODTParameters(
            na=1.2,
            wavelength_m=532e-9,
            img_size_px=64,
            px_size_m=100e-9,
            n_sol=1.33,
            na_illumination=0.9,
        )
        self.config = ODTConfig(
            approx_type="Born",
            hermite_symmetry=True,
            precision=ArrayPrecision(),
            edge_size=0,
        )
        # Calculate appropriate spectrum size based on aperture size
        self.spectrum_size = self.params.aperturesize_px

    def test_mismatched_spectrum_count(self) -> None:
        """Test that ValueError is raised when spectrum counts don't match."""
        # Create mock data with mismatched counts
        cp_spectrums_1 = [
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
        ]
        ref_cp_spectrums_1 = [
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
        ]
        cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]  # Different count
        ref_cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]

        with pytest.raises(ValueError, match="The number of spectrums in both datasets must be the same"):
            calculate_odt_difference(
                cp_spectrums_1,
                ref_cp_spectrums_1,
                cp_spectrums_2,
                ref_cp_spectrums_2,
                self.params,
                self.config,
            )

    def test_mismatched_reference_spectrum_count(self) -> None:
        """Test that ValueError is raised when reference spectrum counts don't match."""
        cp_spectrums_1 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_1 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_2 = [
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
        ]  # Different count

        with pytest.raises(ValueError, match="The number of reference spectrums in both datasets must be the same"):
            calculate_odt_difference(
                cp_spectrums_1,
                ref_cp_spectrums_1,
                cp_spectrums_2,
                ref_cp_spectrums_2,
                self.params,
                self.config,
            )

    def test_mismatched_spectrum_and_reference_count_dataset1(self) -> None:
        """Test that ValueError is raised when spectrum and reference counts don't match in dataset 1."""
        # Make sure the first two checks pass, but the third check fails
        cp_spectrums_1 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_1 = [
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
        ]  # Different count from cp_spectrums_1
        cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_2 = [
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
            jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j,
        ]  # Same count as ref_cp_spectrums_1

        with pytest.raises(ValueError, match="The number of spectrums and reference spectrums must match in dataset 1"):
            calculate_odt_difference(
                cp_spectrums_1,
                ref_cp_spectrums_1,
                cp_spectrums_2,
                ref_cp_spectrums_2,
                self.params,
                self.config,
            )

    def test_function_signature_validation(self) -> None:
        """Test that the function validates input signatures correctly."""
        # Create mock data with correct spectrum sizes
        cp_spectrums_1 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_1 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]
        ref_cp_spectrums_2 = [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j]

        # This should not raise an error during input validation
        # We won't test the full ODT reconstruction due to computational complexity
        # and the need for realistic test data
        try:
            # Only test that function starts without input validation errors
            # We expect it may fail later due to simplified test data
            calculate_odt_difference(
                cp_spectrums_1,
                ref_cp_spectrums_1,
                cp_spectrums_2,
                ref_cp_spectrums_2,
                self.params,
                self.config,
            )
        except ValueError as e:
            # If it's an input validation error we're testing for, re-raise
            if any(keyword in str(e) for keyword in ["number of spectrums", "must be the same", "must match"]):
                raise
            # Otherwise, it's likely a computational error with our simplified test data
            # which is expected and acceptable for this test

    def test_silent_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that difference reconstruction status output is silent by default."""

        def fake_odt(
            _cp_spectrums: Sequence[Array],
            _ref_cp_spectrums: Sequence[Array],
            _params: ODTParameters,
            _config: ODTConfig,
        ) -> tuple[Array, Array]:
            return jnp.ones((1, 1, 1)), jnp.ones((1, 1, 1))

        monkeypatch.setattr(odt_module, "odt", fake_odt)

        calculate_odt_difference(
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            self.params,
            self.config,
        )

        assert not capsys.readouterr().out

    def test_verbose_prints_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test that difference reconstruction status output is opt-in."""

        def fake_odt(
            _cp_spectrums: Sequence[Array],
            _ref_cp_spectrums: Sequence[Array],
            _params: ODTParameters,
            _config: ODTConfig,
        ) -> tuple[Array, Array]:
            return jnp.ones((1, 1, 1)), jnp.ones((1, 1, 1))

        monkeypatch.setattr(odt_module, "odt", fake_odt)
        config = ODTConfig(verbose=True)

        calculate_odt_difference(
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            [jnp.ones((self.spectrum_size, self.spectrum_size)) + 0j],
            self.params,
            config,
        )

        captured = capsys.readouterr().out
        assert "Reconstructing ODT from dataset 1..." in captured
        assert "Reconstructing ODT from dataset 2..." in captured


@pytest.mark.parametrize(
    "ewald_embedding_mode",
    [EwaldEmbeddingMode.TRUNCATE, EwaldEmbeddingMode.NEAREST, EwaldEmbeddingMode.LINEAR],
)
@pytest.mark.parametrize("mode", ["Forward", "Backward"])
def test_synthesize_spectrum_scatter_matches_dense_weight_reference(
    ewald_embedding_mode: EwaldEmbeddingMode,
    mode: str,
) -> None:
    """Test that the scatter-add embedding matches the dense-weight reference."""
    params = ODTParameters(
        na=0.4,
        wavelength_m=1.0,
        img_size_px=32,
        px_size_m=1.0,
        n_sol=1.33,
        na_illumination=0.3,
    )
    config = ODTConfig(hermite_symmetry=False, ewald_embedding_mode=ewald_embedding_mode)
    spectrum_size = 2 * params.aperturesize_px + 1
    shape_3d = (spectrum_size, spectrum_size, params.freq_axial_extent_px)
    coords = jnp.arange(spectrum_size**2, dtype=jnp.float32).reshape(spectrum_size, spectrum_size)
    scattering_spectrums = [
        ScatteringSpectrum(jnp.asarray(coords + 1.0, dtype=jnp.complex64), (0, 0)),
        ScatteringSpectrum(jnp.asarray(coords * 1j, dtype=jnp.complex64), (2, -1), 0.5 + 0.25j),
        ScatteringSpectrum(jnp.asarray(coords - 3.0, dtype=jnp.complex64), (-1, 2)),
    ]

    synthesized = synthesize_spectrum(scattering_spectrums, params, config, mode=mode)

    expected_spectrum = jnp.zeros(shape_3d, dtype=jnp.complex64)
    expected_weight = jnp.zeros(shape_3d, dtype=jnp.float32)
    for scattering_spectrum in scattering_spectrums:
        kz_disk = odt_module._calc_kz_disk(  # noqa: SLF001
            params,
            spectrum_size,
            scattering_spectrum.illumination_vector,
            config.precision,
        )
        embedding_weight = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
            shape_3d,
            params,
            scattering_spectrum.illumination_vector,
            mode,
            ewald_embedding_mode,
        )
        tiled = jnp.stack([scattering_spectrum.array * 2j * kz_disk] * shape_3d[2], axis=2)
        expected_spectrum += scattering_spectrum.coefficient * tiled * embedding_weight
        expected_weight += embedding_weight
    expected = expected_spectrum / jnp.where(expected_weight > 0, expected_weight, 1)

    assert synthesized.shape == expected.shape
    assert bool(jnp.allclose(synthesized, expected, rtol=1e-5, atol=1e-5))


@pytest.mark.parametrize(
    "ewald_embedding_mode",
    [EwaldEmbeddingMode.TRUNCATE, EwaldEmbeddingMode.NEAREST, EwaldEmbeddingMode.LINEAR],
)
def test_synthesize_spectrum_drops_negative_out_of_range_axial_planes(
    ewald_embedding_mode: EwaldEmbeddingMode,
) -> None:
    """Test that negative out-of-range axial indices do not wrap to the last plane."""
    params = ODTParameters(
        na=2 / 3,
        wavelength_m=1.0,
        img_size_px=8,
        px_size_m=0.375,
        n_sol=1.0,
        na_illumination=2 / 3,
    )
    config = ODTConfig(hermite_symmetry=False, ewald_embedding_mode=ewald_embedding_mode)
    spectrum_size = 2 * params.aperturesize_px + 1
    shape_3d = (spectrum_size, spectrum_size, params.freq_axial_extent_px)
    scattering_spectrum = ScatteringSpectrum(
        jnp.ones((spectrum_size, spectrum_size), dtype=jnp.complex64),
        (4, 0),
    )

    synthesized = synthesize_spectrum([scattering_spectrum], params, config, mode="Backward")

    kz_disk = odt_module._calc_kz_disk(  # noqa: SLF001
        params,
        spectrum_size,
        scattering_spectrum.illumination_vector,
        config.precision,
    )
    embedding_weight = odt_module._calc_ewald_embedding_weight(  # noqa: SLF001
        shape_3d,
        params,
        scattering_spectrum.illumination_vector,
        "Backward",
        ewald_embedding_mode,
    )
    expected_spectrum = jnp.stack([scattering_spectrum.array * 2j * kz_disk] * shape_3d[2], axis=2) * embedding_weight
    expected = expected_spectrum / jnp.where(embedding_weight > 0, embedding_weight, 1)

    assert bool(jnp.allclose(synthesized, expected, rtol=1e-5, atol=1e-5))
    assert not bool(jnp.any(synthesized[:, :, -1]))
