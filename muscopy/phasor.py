"""Phasor Analysis for multi-dimensional spectral imaging data.

This module provides:

- `PhasorResult`: A named tuple to store phasor analysis results.
- `PhasorParameters`: A class to store phasor analysis parameters.
- `phasor`: A function to calculate phasor components from spectral imaging data.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

MIN_WAVENUMBER_ELEMENTS = 2


class PhasorResult(NamedTuple):
    r"""Phasor analysis result."""

    g: Array
    """
    g : `jax.Array`
        Real component of phasor (cosine transform).
        Shape matches the spatial dimensions of input array.
    """
    s: Array
    """
    s : `jax.Array`
        Imaginary component of phasor (sine transform).
        Shape matches the spatial dimensions of input array.
        """
    i_sum: Array
    """
    i_sum : `jax.Array`
        Total intensity used for normalization.
        Shape matches the spatial dimensions of input array.
    """


@dataclasses.dataclass
class PhasorParameters:
    r"""Parameters for phasor analysis.

    Attributes
    ----------
    wavenumbers : `jax.Array`
        1D array of wavenumbers corresponding to the spectral dimension.
        Must have at least 2 elements and non-zero range.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from muscopy.phasor import PhasorParameters
    >>> wavenumbers = jnp.linspace(1000, 2000, 100)
    >>> params = PhasorParameters(wavenumbers=wavenumbers)
    >>> params.wavenumber_min
    1000.0
    >>> params.wavenumber_range
    1000.0
    """

    wavenumbers: Array

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        self.verify_parameters()

    def verify_parameters(self) -> None:
        """Verify the parameters.

        Raises
        ------
        ValueError
            1. If wavenumbers is not a 1D array.
            2. If wavenumbers has less than 2 elements.
            3. If wavenumber range is zero (all wavenumbers are identical).
        """
        if self.wavenumbers.ndim != 1:
            msg = f"wavenumbers must be 1D array, got {self.wavenumbers.ndim}D"
            raise ValueError(msg)
        if self.wavenumbers.shape[0] < MIN_WAVENUMBER_ELEMENTS:
            msg = f"wavenumbers must have at least {MIN_WAVENUMBER_ELEMENTS} elements, got {self.wavenumbers.shape[0]}"
            raise ValueError(msg)
        if self.wavenumber_range == 0:
            msg = "wavenumber range cannot be zero (all wavenumbers are identical)"
            raise ValueError(msg)

    @property
    def wavenumber_min(self) -> float:
        """Minimum wavenumber value.

        Returns
        -------
        `float`
            Minimum wavenumber
        """
        return float(jnp.min(self.wavenumbers))

    @property
    def wavenumber_max(self) -> float:
        """Maximum wavenumber value.

        Returns
        -------
        `float`
            Maximum wavenumber
        """
        return float(jnp.max(self.wavenumbers))

    @property
    def wavenumber_range(self) -> float:
        """Range of wavenumbers.

        Returns
        -------
        `float`
            Wavenumber range (max - min)
        """
        return self.wavenumber_max - self.wavenumber_min

    def normalized_frequencies(self) -> Array:
        r"""Calculate normalized frequencies u[i] in range [0, 1].

        The normalized frequency is calculated as:

        .. math::
            u_i = \frac{k_i - k_{min}}{k_{max} - k_{min}}

        Returns
        -------
        `jax.Array`
            Normalized frequencies with shape matching wavenumbers
        """
        return (self.wavenumbers - self.wavenumber_min) / self.wavenumber_range


def phasor(
    array: Array,
    params: PhasorParameters,
    *,
    spectral_axis: int = -1,
) -> PhasorResult:
    r"""Calculate phasor components from spectral imaging data.

    Computes the phasor (g, s) components for multi-dimensional spectral data
    using the following formulas:

    .. math::
        g = \frac{1}{I_{total}} \sum_i I_i \cos(2\pi u_i)

    .. math::
        s = \frac{1}{I_{total}} \sum_i I_i \sin(2\pi u_i)

    where :math:`u_i = \frac{k_i - k_{min}}{k_{max} - k_{min}}` is the normalized frequency.

    The spatial dimensionality is automatically detected from the input array:

    - 3D array (x, y, omega) -> 2D spatial phasor result
    - 4D array (x, y, z, omega) -> 3D spatial phasor result

    Parameters
    ----------
    array : `jax.Array`
        Input spectral data array.
        For 2D spatial data: shape (x, y, omega) or with omega at spectral_axis
        For 3D spatial data: shape (x, y, z, omega) or with omega at spectral_axis
    params : `PhasorParameters`
        Phasor parameters including wavenumbers array
    spectral_axis : `int`, optional
        Axis index of spectral dimension, by default -1 (last axis)

    Returns
    -------
    `PhasorResult`
        Named tuple containing:

        - g: Real component of phasor (cosine transform)
        - s: Imaginary component of phasor (sine transform)
        - i_sum: Total intensity used for normalization

    Raises
    ------
    ValueError
        1. If array is not 3D or 4D.
        2. If spectral dimension size does not match wavenumbers length.
        3. If spectral_axis is out of bounds.

    Examples
    --------
    For 2D spatial data with spectral dimension:

    >>> import jax.numpy as jnp
    >>> from muscopy.phasor import phasor, PhasorParameters
    >>> # Create sample data: 64x64 spatial, 100 spectral points
    >>> data = jnp.ones((64, 64, 100))
    >>> wavenumbers = jnp.linspace(1000, 2000, 100)
    >>> params = PhasorParameters(wavenumbers=wavenumbers)
    >>> result = phasor(data, params)
    >>> result.g.shape
    (64, 64)

    For 3D spatial data:

    >>> data_3d = jnp.ones((32, 32, 16, 100))
    >>> result_3d = phasor(data_3d, params)
    >>> result_3d.g.shape
    (32, 32, 16)
    """
    # Validate parameters
    params.verify_parameters()

    # Validate array dimensions (auto-detect 2D or 3D spatial)
    if array.ndim not in {3, 4}:
        msg = f"array must be 3D or 4D, got {array.ndim}D"
        raise ValueError(msg)

    # Normalize spectral_axis to positive index
    if spectral_axis < 0:
        spectral_axis = array.ndim + spectral_axis

    if spectral_axis < 0 or spectral_axis >= array.ndim:
        msg = f"spectral_axis {spectral_axis} is out of bounds for array with {array.ndim} dimensions"
        raise ValueError(msg)

    # Validate spectral dimension matches wavenumbers
    spectral_size = array.shape[spectral_axis]
    if spectral_size != params.wavenumbers.shape[0]:
        msg = (
            f"Spectral dimension size ({spectral_size}) does not match "
            f"wavenumbers length ({params.wavenumbers.shape[0]})"
        )
        raise ValueError(msg)

    # Get normalized frequencies
    u = params.normalized_frequencies()  # shape: (omega,)

    # Compute 2*pi*u for trigonometric functions
    two_pi_u = 2 * jnp.pi * u

    # Reshape for broadcasting across spatial dimensions
    broadcast_shape = [1] * array.ndim
    broadcast_shape[spectral_axis] = -1
    two_pi_u = two_pi_u.reshape(broadcast_shape)

    # Calculate total intensity (sum along spectral axis)
    i_sum = jnp.sum(array, axis=spectral_axis)

    # Avoid division by zero using jnp.where
    i_sum_safe = jnp.where(i_sum == 0, 1.0, i_sum)

    # Calculate g component (cosine)
    g_unnorm = jnp.sum(array * jnp.cos(two_pi_u), axis=spectral_axis)
    g = g_unnorm / i_sum_safe

    # Calculate s component (sine)
    s_unnorm = jnp.sum(array * jnp.sin(two_pi_u), axis=spectral_axis)
    s = s_unnorm / i_sum_safe

    # Set zero intensity pixels to zero phasor
    g = jnp.where(i_sum == 0, 0.0, g)
    s = jnp.where(i_sum == 0, 0.0, s)

    return PhasorResult(g=g, s=s, i_sum=i_sum)
