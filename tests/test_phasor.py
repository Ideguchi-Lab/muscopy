"""Test cases for phasor module."""

import jax.numpy as jnp
import pytest

from muscopy.phasor import PhasorParameters, PhasorResult, phasor

# Test PhasorParameters


def test_phasor_parameters_valid() -> None:
    """Test valid parameter initialization."""
    wavenumbers = jnp.linspace(1000, 2000, 100)
    params = PhasorParameters(wavenumbers=wavenumbers)
    assert params.wavenumber_min == 1000.0
    assert params.wavenumber_max == 2000.0
    assert params.wavenumber_range == 1000.0


def test_phasor_parameters_normalized_frequencies() -> None:
    """Test normalized frequency calculation."""
    wavenumbers = jnp.array([1000.0, 1500.0, 2000.0])
    params = PhasorParameters(wavenumbers=wavenumbers)
    u = params.normalized_frequencies()
    expected = jnp.array([0.0, 0.5, 1.0])
    assert jnp.allclose(u, expected)


def test_phasor_parameters_invalid_not_1d() -> None:
    """Test error for non-1D wavenumbers."""
    wavenumbers = jnp.ones((10, 10))
    with pytest.raises(ValueError, match="must be 1D array"):
        PhasorParameters(wavenumbers=wavenumbers)


def test_phasor_parameters_invalid_too_short() -> None:
    """Test error for wavenumbers with less than 2 elements."""
    wavenumbers = jnp.array([1000.0])
    with pytest.raises(ValueError, match="at least 2 elements"):
        PhasorParameters(wavenumbers=wavenumbers)


def test_phasor_parameters_invalid_zero_range() -> None:
    """Test error for wavenumbers with zero range."""
    wavenumbers = jnp.array([1000.0, 1000.0, 1000.0])
    with pytest.raises(ValueError, match="range cannot be zero"):
        PhasorParameters(wavenumbers=wavenumbers)


# Test 2D spatial phasor analysis


def test_phasor_2d_output_shape() -> None:
    """Test output shape for 2D spatial data."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((64, 64, 50))
    result = phasor(data, params)
    assert isinstance(result, PhasorResult)
    assert result.g.shape == (64, 64)
    assert result.s.shape == (64, 64)
    assert result.i_sum.shape == (64, 64)


def test_phasor_2d_output_type() -> None:
    """Test that result is a PhasorResult named tuple."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((32, 32, 50))
    result = phasor(data, params)
    assert isinstance(result, PhasorResult)
    assert hasattr(result, "g")
    assert hasattr(result, "s")
    assert hasattr(result, "i_sum")


def test_phasor_2d_constant_spectrum() -> None:
    """Test phasor for constant spectrum."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((32, 32, 50))
    result = phasor(data, params)
    # For constant spectrum, i_sum should be spectral_size
    assert jnp.allclose(result.i_sum, 50.0)


def test_phasor_2d_zero_intensity() -> None:
    """Test handling of zero intensity pixels."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.zeros((16, 16, 50))
    result = phasor(data, params)
    assert jnp.all(result.g == 0)
    assert jnp.all(result.s == 0)
    assert jnp.all(result.i_sum == 0)


def test_phasor_2d_spectral_axis_first() -> None:
    """Test spectral_axis at first position."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((50, 32, 32))  # spectral axis first
    result = phasor(data, params, spectral_axis=0)
    assert result.g.shape == (32, 32)
    assert result.s.shape == (32, 32)
    assert result.i_sum.shape == (32, 32)


def test_phasor_2d_spectral_axis_middle() -> None:
    """Test spectral_axis in the middle."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((32, 50, 32))  # spectral axis in middle
    result = phasor(data, params, spectral_axis=1)
    assert result.g.shape == (32, 32)
    assert result.s.shape == (32, 32)
    assert result.i_sum.shape == (32, 32)


def test_phasor_invalid_dimensions_2d() -> None:
    """Test error for 2D array (missing spectral dimension)."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((64, 64))
    with pytest.raises(ValueError, match="must be 3D or 4D"):
        phasor(data, params)


def test_phasor_invalid_dimensions_5d() -> None:
    """Test error for 5D array."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((32, 32, 16, 8, 50))
    with pytest.raises(ValueError, match="must be 3D or 4D"):
        phasor(data, params)


def test_phasor_spectral_size_mismatch() -> None:
    """Test error for spectral dimension mismatch."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((64, 64, 100))  # 100 != 50 wavenumbers
    with pytest.raises(ValueError, match="does not match wavenumbers length"):
        phasor(data, params)


# Test 3D spatial phasor analysis


def test_phasor_3d_output_shape() -> None:
    """Test output shape for 3D spatial data."""
    wavenumbers = jnp.linspace(1000, 2000, 30)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((32, 32, 16, 30))
    result = phasor(data, params)
    assert isinstance(result, PhasorResult)
    assert result.g.shape == (32, 32, 16)
    assert result.s.shape == (32, 32, 16)
    assert result.i_sum.shape == (32, 32, 16)


def test_phasor_3d_spectral_axis_first() -> None:
    """Test spectral_axis at first position for 3D spatial data."""
    wavenumbers = jnp.linspace(1000, 2000, 30)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((30, 16, 16, 8))  # spectral axis first
    result = phasor(data, params, spectral_axis=0)
    assert result.g.shape == (16, 16, 8)


def test_phasor_3d_spectral_axis_middle() -> None:
    """Test spectral_axis in the middle for 3D spatial data."""
    wavenumbers = jnp.linspace(1000, 2000, 30)
    params = PhasorParameters(wavenumbers=wavenumbers)
    data = jnp.ones((16, 30, 16, 8))  # spectral axis at position 1
    result = phasor(data, params, spectral_axis=1)
    assert result.g.shape == (16, 16, 8)


# Numerical validation tests


def test_phasor_single_peak_spectrum() -> None:
    """Test phasor for spectrum with single peak at known position."""
    # Create wavenumbers from 0 to 1
    wavenumbers = jnp.linspace(0, 1, 100)
    params = PhasorParameters(wavenumbers=wavenumbers)

    # Create a Gaussian peak at u=0.25
    peak_u = 0.25
    u = params.normalized_frequencies()
    spectrum = jnp.exp(-((u - peak_u) ** 2) / (2 * 0.01**2))

    # Make it 3D (1x1 spatial)
    data = spectrum.reshape(1, 1, -1)

    result = phasor(data, params)

    # For a narrow peak at u=0.25:
    # g should be approximately cos(2*pi*0.25) = 0
    # s should be approximately sin(2*pi*0.25) = 1
    assert jnp.abs(result.g[0, 0]) < 0.15
    assert result.s[0, 0] > 0.85


def test_phasor_magnitude_bounded() -> None:
    """Test that phasor magnitude is bounded by 1."""
    wavenumbers = jnp.linspace(1000, 2000, 50)
    params = PhasorParameters(wavenumbers=wavenumbers)

    # Positive spectrum
    data = jnp.abs(jnp.sin(jnp.linspace(0, 10, 50))).reshape(1, 1, -1) + 0.1

    result = phasor(data, params)

    # Phasor magnitude should be <= 1
    magnitude = jnp.sqrt(result.g**2 + result.s**2)
    assert jnp.all(magnitude <= 1.0 + 1e-6)


def test_phasor_phase_recovery() -> None:
    """Test that phasor angle corresponds to spectral center of mass."""
    # Create wavenumbers from 0 to 1
    wavenumbers = jnp.linspace(0, 1, 100)
    params = PhasorParameters(wavenumbers=wavenumbers)

    # Test with different peak positions
    for peak_u in [0.0, 0.25, 0.5, 0.75]:
        u = params.normalized_frequencies()
        spectrum = jnp.exp(-((u - peak_u) ** 2) / (2 * 0.05**2))
        data = spectrum.reshape(1, 1, -1)

        result = phasor(data, params)

        # Phasor angle should approximately match 2*pi*peak_u
        phasor_angle = jnp.arctan2(result.s[0, 0], result.g[0, 0])
        expected_angle = 2 * jnp.pi * peak_u

        # Normalize angles to [0, 2*pi) for comparison
        phasor_angle = phasor_angle % (2 * jnp.pi)  # noqa: PLR6104
        expected_angle = expected_angle % (2 * jnp.pi)  # noqa: PLR6104

        # Allow some tolerance due to Gaussian width
        angle_diff = jnp.abs(phasor_angle - expected_angle)
        angle_diff = jnp.minimum(angle_diff, 2 * jnp.pi - angle_diff)
        assert angle_diff < 0.3, f"Failed for peak_u={peak_u}"


def test_phasor_mixed_zero_nonzero_pixels() -> None:
    """Test handling of mixed zero and non-zero intensity pixels."""
    wavenumbers = jnp.linspace(1000, 2000, 20)
    params = PhasorParameters(wavenumbers=wavenumbers)

    # Create data with some zero and some non-zero pixels
    data = jnp.zeros((4, 4, 20))
    data = data.at[0, 0, :].set(1.0)  # Non-zero pixel at (0,0)
    data = data.at[2, 2, :].set(2.0)  # Non-zero pixel at (2,2)

    result = phasor(data, params)

    # Zero intensity pixels should have zero phasor
    assert result.g[1, 1] == 0.0
    assert result.s[1, 1] == 0.0

    # Non-zero intensity pixels should have non-zero i_sum
    assert result.i_sum[0, 0] == 20.0
    assert result.i_sum[2, 2] == 40.0
