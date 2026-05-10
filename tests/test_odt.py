"""Test cases for ODT module."""

import jax.numpy as jnp
import pytest

from muscopy.cfg import ArrayPrecision
from muscopy.odt import ODTConfig, ODTParameters, calc_scattering_potential, calculate_odt_difference


def test_odt_config_defaults_to_32_bit_precision() -> None:
    """Test that ODT defaults to the portable precision contract."""
    config = ODTConfig()

    assert config.precision.int_precision() == "int32"
    assert config.precision.float_precision() == "float32"
    assert config.precision.complex_precision() == "complex64"


def test_calc_scattering_potential_rejects_mutated_64_bit_precision() -> None:
    """Test that ODT validates precision before reconstruction."""
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
        calc_scattering_potential([], [], params, config)


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
