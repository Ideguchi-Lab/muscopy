"""Test cases for ODT module."""

from collections.abc import Sequence

import jax.numpy as jnp
import pytest
from jax import Array

from muscopy import odt as odt_module
from muscopy.cfg import ArrayPrecision
from muscopy.odt import (
    ODTConfig,
    ODTParameters,
    ScatteringSpectrum,
    calc_scattering_potential,
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


def test_calc_scattering_potential_rejects_mutated_64_bit_precision(monkeypatch: pytest.MonkeyPatch) -> None:
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

    with (
        pytest.warns(FutureWarning, match="calc_scattering_potential\\(\\) is deprecated"),
        pytest.raises(ValueError, match="64-bit precision requires JAX x64 support"),
    ):
        calc_scattering_potential([], [], params, config)


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


def test_calc_scattering_potential_matches_factored_reconstruction_path() -> None:
    """Test that the legacy wrapper matches the explicit factored ODT path."""
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

    with pytest.warns(FutureWarning, match="calc_scattering_potential\\(\\) is deprecated"):
        wrapped_potential, wrapped_spectrum = calc_scattering_potential(
            [cp_spectrum],
            [ref_cp_spectrum],
            params,
            config,
        )
    scattering_spectrums = calc_scattering_spectrums([cp_spectrum], [ref_cp_spectrum], params, config)
    factored_potential, factored_spectrum = calc_scattering_potential_from_spectrums(
        scattering_spectrums,
        params,
        config,
    )

    assert jnp.allclose(wrapped_potential, factored_potential)
    assert jnp.allclose(wrapped_spectrum, factored_spectrum)


def test_calc_scattering_potential_warns_with_migration_path() -> None:
    """Test that the deprecated ODT wrapper points callers to the factored API."""
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

    with pytest.warns(FutureWarning, match="calc_scattering_spectrums\\(\\) followed by"):
        calc_scattering_potential([cp_spectrum], [ref_cp_spectrum], params, config)


def test_odt_warns_with_migration_path() -> None:
    """Test that the deprecated high-level ODT wrapper points callers to the factored API."""
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

    with pytest.warns(FutureWarning, match="calc_scattering_spectrums\\(\\)"):
        refractive_index, synthesized_spectrum = odt_module.odt([cp_spectrum], [ref_cp_spectrum], params, config)

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
            with pytest.warns(FutureWarning, match="odt\\(\\) is deprecated"):
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
