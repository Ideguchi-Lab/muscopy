"""Digital hologram generation for MLB simulation results.

This module provides:

- `HologramGenerator`: Digital hologram generation class
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from .mlb import MLBParameters

# Constants for readability
_DEFAULT_BIT_DEPTH = 12
_NDIM_2D = 2


class HologramGenerator:
    """Digital hologram generator for MLB simulation results.

    This class generates digital holograms by interfering the scattered field
    with a reference beam, simulating off-axis digital holography.
    """

    def __init__(self, params: MLBParameters) -> None:
        """Initialize hologram generator.

        Parameters
        ----------
        params : `MLBParameters`
            MLB simulation parameters
        """
        self.params = params
        self._target_field: Array | None = None
        self._target_field_fft: Array | None = None
        self._offaxis_position: tuple[int, int] = (0, 0)

    def set_target_field(self, field: Array) -> None:
        """Set the target field for hologram generation.

        Parameters
        ----------
        field : `Array`
            2D target electromagnetic field

        Raises
        ------
        ValueError
            If field is not 2D
        """
        if len(field.shape) != _NDIM_2D:
            msg = "Target field must be 2D"
            raise ValueError(msg)

        # Normalize the field
        field_normalized = field / jnp.sqrt(jnp.sum(jnp.abs(field) ** 2))
        self._target_field = field_normalized
        self._target_field_fft = jnp.fft.fftshift(jnp.fft.fft2(field_normalized))

    def set_offaxis_position(self, offset_x: int, offset_y: int) -> None:
        """Set the off-axis reference beam position.

        Parameters
        ----------
        offset_x : `int`
            x-offset in pixels
        offset_y : `int`
            y-offset in pixels
        """
        self._offaxis_position = (offset_x, offset_y)

    def generate_hologram(
        self,
        hologram_shape: tuple[int, int],
        reference_amplitude: float = 1.0,
        bit_depth: int = _DEFAULT_BIT_DEPTH,
        output_dtype: str = "int16",
    ) -> Array:
        r"""Generate digital hologram.

        Parameters
        ----------
        hologram_shape : `tuple`\[`int`, `int`\]
            Shape of the output hologram
        reference_amplitude : `float`, default=1.0
            Amplitude of the reference beam relative to target field
        bit_depth : `int`, default=12
            Bit depth for quantization
        output_dtype : `str`, default='int16'
            Output data type

        Returns
        -------
        `Array`
            Digital hologram as intensity image

        Raises
        ------
        ValueError
            If target field is not set
        """
        if self._target_field_fft is None:
            msg = "Target field not set. Call set_target_field() first."
            raise ValueError(msg)

        # Create reference beam
        reference_fft = jnp.zeros(hologram_shape, dtype=jnp.complex64)
        offset_x, offset_y = self._offaxis_position
        reference_fft = reference_fft.at[offset_x, offset_y].set(1)

        reference_field = jnp.fft.ifft2(jnp.fft.ifftshift(reference_fft))
        # Normalize reference field
        reference_field = reference_field / jnp.sqrt(jnp.sum(jnp.abs(reference_field) ** 2)) * reference_amplitude

        # Pad target field to hologram size
        target_fft_padded = jnp.zeros(hologram_shape, dtype=jnp.complex64)
        h_start = hologram_shape[0] // 2 - self.params.xy_shape[0] // 2
        h_end = h_start + self.params.xy_shape[0]
        w_start = hologram_shape[1] // 2 - self.params.xy_shape[1] // 2
        w_end = w_start + self.params.xy_shape[1]

        target_fft_padded = target_fft_padded.at[h_start:h_end, w_start:w_end].set(self._target_field_fft)
        target_field = jnp.fft.ifft2(jnp.fft.ifftshift(target_fft_padded))

        # Generate hologram by interference
        total_field = reference_field + target_field
        intensity = jnp.abs(total_field) ** 2

        return self._convert_to_integer(intensity, bit_depth, output_dtype)

    @staticmethod
    def _convert_to_integer(
        image: Array,
        bit_depth: int,
        dtype: str,
    ) -> Array:
        """Convert floating point image to integer representation.

        Parameters
        ----------
        image : `jax.Array`
            Input floating point image
        bit_depth : `int`
            Number of bits for quantization
        dtype : `str`
            Output data type

        Returns
        -------
        `jax.Array`
            Quantized integer image
        """
        image_normalized = image / jnp.max(image)

        # Quantize
        max_value = 2**bit_depth - 1
        image_quantized = image_normalized * max_value

        return jnp.asarray(image_quantized.astype(dtype))
