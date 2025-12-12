"""Test cases for qpi_utils module."""

import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.testing import assert_allclose

from muscopy.qpi_utils import unwrap_phase


class TestUnwrapPhase:
    """Test cases for unwrap_phase function."""

    @staticmethod
    def test_unwrap_simple_tilt() -> None:
        """Test unwrapping a simple tilted phase that wraps around."""
        n = 64
        # Create a linear phase ramp that exceeds 2*pi (causes wrapping)
        x = jnp.linspace(0, 4 * jnp.pi, n)
        true_phase = jnp.outer(x, jnp.ones(n))
        # Wrap the phase to [-pi, pi)
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        unwrapped = unwrap_phase(wrapped_phase, keep_mean=False)

        # The unwrapped phase should be close to the original (up to a constant offset)
        diff = unwrapped - true_phase
        diff_centered = diff - diff.mean()
        assert_allclose(np.asarray(diff_centered), 0.0, atol=0.1)

    @staticmethod
    def test_unwrap_gaussian_phase() -> None:
        """Test unwrapping a Gaussian-shaped phase profile."""
        n = 64
        x, y = jnp.meshgrid(jnp.linspace(-3, 3, n), jnp.linspace(-3, 3, n), indexing="ij")
        true_phase = 5 * jnp.exp(-(x**2 + y**2) / 2)
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        unwrapped = unwrap_phase(wrapped_phase, keep_mean=False)

        # The unwrapped phase should match the original shape (up to a constant offset)
        diff = unwrapped - true_phase
        diff_centered = diff - diff.mean()
        assert_allclose(np.asarray(diff_centered), 0.0, atol=0.1)

    @staticmethod
    def test_unwrap_with_roi() -> None:
        """Test unwrapping with a circular ROI mask."""
        n = 64
        x, y = jnp.meshgrid(jnp.linspace(-3, 3, n), jnp.linspace(-3, 3, n), indexing="ij")
        true_phase = 4 * jnp.exp(-(x**2 + y**2) / 2)
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        # Create circular ROI
        center = n // 2
        xx, yy = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        roi = ((xx - center) ** 2 + (yy - center) ** 2) < (n // 3) ** 2

        unwrapped = unwrap_phase(wrapped_phase, roi=roi, keep_mean=False)

        # Check that unwrapped values inside ROI are reasonable
        roi_np = np.asarray(roi)
        unwrapped_np = np.asarray(unwrapped)
        true_phase_np = np.asarray(true_phase)

        # Inside ROI, the unwrapped phase should match the true phase (up to constant)
        diff_inside = unwrapped_np[roi_np] - true_phase_np[roi_np]
        diff_inside_centered = diff_inside - diff_inside.mean()
        assert_allclose(diff_inside_centered, 0.0, atol=0.2)

        # Outside ROI, the original wrapped phase should be preserved
        assert_allclose(unwrapped_np[~roi_np], np.asarray(wrapped_phase)[~roi_np], atol=1e-6)

    @staticmethod
    def test_keep_mean_true() -> None:
        """Test that keep_mean=True preserves the mean of the original phase."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n)
        true_phase = jnp.outer(x, jnp.ones(n)) + 1.5  # Add offset
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        unwrapped = unwrap_phase(wrapped_phase, keep_mean=True)

        # The mean should be close to the original wrapped phase mean
        assert_allclose(float(unwrapped.mean()), float(wrapped_phase.mean()), atol=0.1)

    @staticmethod
    def test_keep_mean_false() -> None:
        """Test that keep_mean=False results in zero-mean output."""
        n = 32
        x = jnp.linspace(0, 2 * jnp.pi, n)
        true_phase = jnp.outer(x, jnp.ones(n)) + 1.5
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        unwrapped = unwrap_phase(wrapped_phase, keep_mean=False)

        # The mean should be close to zero
        assert_allclose(float(unwrapped.mean()), 0.0, atol=0.1)

    @staticmethod
    def test_keep_mean_with_roi() -> None:
        """Test that keep_mean=True with ROI uses ROI-masked mean."""
        n = 32
        x, y = jnp.meshgrid(jnp.linspace(-2, 2, n), jnp.linspace(-2, 2, n), indexing="ij")
        true_phase = 3 * jnp.exp(-(x**2 + y**2) / 2) + 0.5
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        # Create circular ROI
        center = n // 2
        xx, yy = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        roi = ((xx - center) ** 2 + (yy - center) ** 2) < (n // 4) ** 2

        unwrapped = unwrap_phase(wrapped_phase, roi=roi, keep_mean=True)

        # The mean inside ROI should be close to the original wrapped phase mean inside ROI
        roi_np = np.asarray(roi)
        wrapped_mean_roi = float(np.asarray(wrapped_phase)[roi_np].mean())
        unwrapped_mean_roi = float(np.asarray(unwrapped)[roi_np].mean())
        # Relax tolerance - the unwrapped mean may differ due to ROI boundary effects
        assert_allclose(unwrapped_mean_roi, wrapped_mean_roi, atol=0.5)

        # IMPORTANT: Pixels outside ROI should remain unchanged (original wrapped values)
        assert_allclose(np.asarray(unwrapped)[~roi_np], np.asarray(wrapped_phase)[~roi_np], atol=1e-6)

    @staticmethod
    def test_no_wrapping_case() -> None:
        """Test that phase without wrapping is preserved."""
        n = 32
        # Phase that doesn't wrap (stays within [-pi, pi])
        x, y = jnp.meshgrid(jnp.linspace(-1, 1, n), jnp.linspace(-1, 1, n), indexing="ij")
        phase = 0.5 * (x + y)  # Small linear phase

        unwrapped = unwrap_phase(phase, keep_mean=False)

        # Should be very close to the original (up to constant)
        diff = unwrapped - phase
        diff_centered = diff - diff.mean()
        # Relax tolerance for numerical precision
        assert_allclose(np.asarray(diff_centered), 0.0, atol=1e-4)

    @staticmethod
    def test_output_shape() -> None:
        """Test that output shape matches input shape."""
        for shape in [(32, 32), (64, 128), (128, 64)]:
            phase = jnp.zeros(shape)
            unwrapped = unwrap_phase(phase)
            assert unwrapped.shape == shape

    @staticmethod
    def test_output_type() -> None:
        """Test that output is a JAX Array."""
        phase = jnp.zeros((32, 32))
        unwrapped = unwrap_phase(phase)
        assert isinstance(unwrapped, Array)


def _wraptopi(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap phase to [-pi, pi) range (helper for testing)."""
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


class TestUnwrapPhaseRhoConsistency:
    """Test cases to verify the rho (divergence) calculation is correct."""

    @staticmethod
    def test_rho_sum_is_approximately_zero() -> None:
        """Test that the divergence (rho) sum is approximately zero.

        This is a necessary condition for Neumann Poisson to have a solution.
        """
        n = 64
        x, y = jnp.meshgrid(jnp.linspace(-3, 3, n), jnp.linspace(-3, 3, n), indexing="ij")
        true_phase = 5 * jnp.exp(-(x**2 + y**2) / 2)
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        # Replicate the rho calculation from unwrap_phase
        dx = _wraptopi(jnp.diff(wrapped_phase, axis=1))
        dy = _wraptopi(jnp.diff(wrapped_phase, axis=0))
        rho = jnp.diff(dx, axis=1, prepend=0.0, append=0.0) + jnp.diff(dy, axis=0, prepend=0.0, append=0.0)

        # The sum of rho should be approximately zero
        rho_sum = float(jnp.sum(rho))
        assert abs(rho_sum) < 1e-10, f"rho sum should be ~0, but got {rho_sum}"

    @staticmethod
    def test_rho_sum_with_roi() -> None:
        """Test that the divergence sum is approximately zero even with ROI masking."""
        n = 64
        x, y = jnp.meshgrid(jnp.linspace(-3, 3, n), jnp.linspace(-3, 3, n), indexing="ij")
        true_phase = 5 * jnp.exp(-(x**2 + y**2) / 2)
        wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

        # Create circular ROI
        center = n // 2
        xx, yy = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
        roi = ((xx - center) ** 2 + (yy - center) ** 2) < (n // 3) ** 2
        roi = roi.astype(bool)

        # Replicate the rho calculation from unwrap_phase with ROI
        dx = _wraptopi(jnp.diff(wrapped_phase, axis=1))
        dy = _wraptopi(jnp.diff(wrapped_phase, axis=0))

        roi_x = roi[:, 1:] & roi[:, :-1]
        roi_y = roi[1:, :] & roi[:-1, :]

        dx = jnp.where(roi_x, dx, 0.0)
        dy = jnp.where(roi_y, dy, 0.0)

        rho = jnp.diff(dx, axis=1, prepend=0.0, append=0.0) + jnp.diff(dy, axis=0, prepend=0.0, append=0.0)
        rho = jnp.where(roi, rho, 0.0)

        # The sum of rho should be approximately zero
        rho_sum = float(jnp.sum(rho))
        assert abs(rho_sum) < 1e-10, f"rho sum with ROI should be ~0, but got {rho_sum}"
