"""Multi-layer Born (MLB) simulation for optical diffraction tomography.

This module provides:

- `MLBParameters`: A dataclass that encapsulates the parameters required for MLB simulation
- `get_scatter_potential`: Convert refractive index to scattering potential
- `get_refractive_index`: Convert scattering potential to refractive index
- `get_fx_fy_phaser`: Generate frequency domain coordinates
- `get_oblique_wave`: Generate oblique illumination wave
- `get_fz_phaser`: Generate axial frequency components
- `get_propagator`: Generate propagation kernel
- `get_green_func`: Generate Green's function
- `get_aperture`: Generate numerical aperture mask
- `propagate`: Propagate field through distance
- `MLBForward`: Forward scattering simulation class
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
from jax import Array
from tqdm import tqdm

# Constants for readability
_EVANESCENT_THRESHOLD = 1e6
_NDIM_2D = 2
_NDIM_3D = 3


@dataclasses.dataclass
class MLBParameters:
    r"""Parameters for Multi-layer Born (MLB) simulation.

    This dataclass encapsulates all parameters required for MLB forward scattering
    simulation including optical parameters, grid dimensions, and precision settings.

    Attributes
    ----------
    wavelength_m : `float`
        Wavelength of light in meters
    numerical_aperture : `float`
        Collection numerical aperture of the objective lens
    n_medium : `float`
        Refractive index of the surrounding medium
    xy_shape : `tuple`\[`int`, `int`\]
        Size of the 2D array (height, width)
    num_layers : `int`
        Number of axial layers in the simulation
    dxy_m : `float`
        Lateral pixel size in meters
    dz_m : `float`
        Axial layer spacing in meters
    """

    wavelength_m: float
    numerical_aperture: float
    n_medium: float
    xy_shape: tuple[int, int]
    num_layers: int
    dxy_m: float
    dz_m: float

    @property
    def k_per_pixel(self) -> float:
        """Wave number per pixel (computed automatically)."""
        return 2 * np.pi / (self.dxy_m * self.xy_shape[0])


def get_scatter_potential(
    params: MLBParameters,
    refractive_index: Array,
) -> Array:
    """Convert refractive index distribution to scattering potential.

    The scattering potential is computed as:
    V(r) = -k_m^2 * (n(r)^2/n_m^2 - 1)

    where k_m = 2π * n_m / λ is the wave number in the medium.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    refractive_index : `jax.Array`
        3D refractive index distribution

    Returns
    -------
    `jax.Array`
        3D scattering potential distribution
    """
    k_medium = 2 * jnp.pi * params.n_medium / params.wavelength_m
    return -(k_medium**2) * (refractive_index**2 / params.n_medium**2 - 1)


def get_refractive_index(
    params: MLBParameters,
    scatter_potential: Array,
) -> Array:
    """Convert scattering potential to refractive index distribution.

    This is the inverse operation of get_scatter_potential.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    scatter_potential : `jax.Array`
        3D scattering potential distribution

    Returns
    -------
    `jax.Array`
        3D refractive index distribution
    """
    k_medium = 2 * jnp.pi * params.n_medium / params.wavelength_m
    return jnp.sqrt(1 - scatter_potential / k_medium**2) * params.n_medium


def get_fx_fy_phaser(params: MLBParameters) -> tuple[Array, Array]:
    r"""Generate frequency domain coordinate grids.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters

    Returns
    -------
    `tuple`\[`jax.Array`, `jax.Array`\]
        Frequency coordinate grids (fx, fy) in 1/m
    """
    fx = jnp.linspace(-0.5 / params.dxy_m, +0.5 / params.dxy_m, params.xy_shape[0])
    fy = jnp.linspace(-0.5 / params.dxy_m, +0.5 / params.dxy_m, params.xy_shape[1])
    fx_phaser, fy_phaser = jnp.meshgrid(fx, fy, indexing="ij")
    return fx_phaser, fy_phaser


def get_oblique_wave(
    params: MLBParameters,
    kx: float,
    ky: float,
) -> Array:
    """Generate oblique illumination wave in spatial domain.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    kx : `float`
        x-component of wave vector in 1/m
    ky : `float`
        y-component of wave vector in 1/m

    Returns
    -------
    `jax.Array`
        2D oblique wave field
    """
    oblique_wave_fft = get_oblique_wave_fft(params, kx, ky)
    return jnp.fft.ifft2(jnp.fft.ifftshift(oblique_wave_fft))


def get_oblique_wave_fft(
    params: MLBParameters,
    kx: float,
    ky: float,
) -> Array:
    """Generate oblique illumination wave in frequency domain.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    kx : `float`
        x-component of wave vector in 1/m
    ky : `float`
        y-component of wave vector in 1/m

    Returns
    -------
    `jax.Array`
        2D oblique wave field in frequency domain
    """
    oblique_wave_fft = jnp.zeros(params.xy_shape, dtype=jnp.complex64)

    # Calculate pixel indices for the wave vector
    kx_idx = int(kx / params.k_per_pixel) + params.xy_shape[0] // 2
    ky_idx = int(ky / params.k_per_pixel) + params.xy_shape[1] // 2

    # Set delta function at the specified frequency
    return oblique_wave_fft.at[kx_idx, ky_idx].set(1)


def get_fz_phaser(params: MLBParameters) -> Array:
    """Generate axial frequency components.

    Computes fz = sqrt((n_m/λ)² - fx² - fy²) with proper handling of
    evanescent waves.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters

    Returns
    -------
    `jax.Array`
        2D array of axial frequency components
    """
    fx_phaser, fy_phaser = get_fx_fy_phaser(params)

    # Axial frequency component
    fz_squared = (params.n_medium / params.wavelength_m) ** 2 - fx_phaser**2 - fy_phaser**2
    fz_phaser = jnp.sqrt(jnp.real(jnp.maximum(fz_squared, 0)))

    # Handle evanescent waves (set to zero where fz would be imaginary)
    evanescent_mask = 1 / fz_phaser > _EVANESCENT_THRESHOLD
    return jnp.where(evanescent_mask, 0, fz_phaser)


def get_propagator(
    params: MLBParameters,
    distance_m: float = 1.0,
) -> Array:
    """Generate free-space propagation kernel.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    distance_m : `float`, default=1.0
        Propagation distance in meters

    Returns
    -------
    `jax.Array`
        2D propagation kernel in frequency domain
    """
    fz_phaser = get_fz_phaser(params)
    propagator = jnp.exp(1j * 2 * jnp.pi * distance_m * fz_phaser)

    # Apply numerical aperture mask
    aperture = get_aperture(params)
    propagator *= aperture
    return propagator


def get_green_func(
    params: MLBParameters,
    distance_m: float = 1.0,
) -> Array:
    """Generate Green's function for scattering.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters
    distance_m : `float`, default=1.0
        Distance in meters

    Returns
    -------
    `jax.Array`
        2D Green's function in frequency domain

    Raises
    ------
    ValueError
        If distance is negative
    """
    if distance_m < 0:
        msg = "Distance must be non-negative"
        raise ValueError(msg)

    fz_phaser = get_fz_phaser(params)
    propagator = jnp.exp(1j * 2 * jnp.pi * distance_m * fz_phaser)

    # Avoid division by zero
    nonzero_fz = fz_phaser != 0
    safe_fz = jnp.where(nonzero_fz, fz_phaser, 1)

    return (-1j * propagator / (4 * jnp.pi) / safe_fz) * nonzero_fz


def get_aperture(params: MLBParameters) -> Array:
    """Generate numerical aperture mask.

    Parameters
    ----------
    params : `MLBParameters`
        MLB simulation parameters

    Returns
    -------
    `jax.Array`
        2D binary aperture mask
    """
    fx_phaser, fy_phaser = get_fx_fy_phaser(params)
    max_spatial_freq = params.numerical_aperture / params.wavelength_m
    return (fx_phaser**2 + fy_phaser**2) < max_spatial_freq**2


def propagate(
    field: Array,
    distance_px: int,
    params: MLBParameters,
) -> Array:
    """Propagate electromagnetic field through free space.

    Parameters
    ----------
    field : `jax.Array`
        2D electromagnetic field
    distance_px : `int`
        Propagation distance in pixels
    params : `MLBParameters`
        MLB simulation parameters

    Returns
    -------
    `jax.Array`
        Propagated electromagnetic field
    """
    distance_m = distance_px * params.dz_m
    propagator = get_propagator(params, distance_m)

    # Forward FFT -> multiply by propagator -> inverse FFT
    field_fft = jnp.fft.fftshift(jnp.fft.fft2(field))
    field_propagated_fft = field_fft * propagator
    return jnp.fft.ifft2(jnp.fft.ifftshift(field_propagated_fft))


class MLBForward:
    """Multi-layer Born forward scattering simulator.

    This class implements the forward model for multi-layer Born scattering,
    which is used to simulate light propagation through a 3D scattering medium.
    """

    def __init__(self, params: MLBParameters) -> None:
        """Initialize MLB forward simulator.

        Parameters
        ----------
        params : `MLBParameters`
            MLB simulation parameters
        """
        self.params = params
        self._input_field_fft: Array | None = None
        self._scattering_potential: Array | None = None

    def set_input_field_fft(self, field_fft: Array) -> None:
        """Set the input illumination field in frequency domain.

        Parameters
        ----------
        field_fft : `jax.Array`
            2D input field in frequency domain

        Raises
        ------
        ValueError
            If field is not 2D
        """
        if len(field_fft.shape) != _NDIM_2D:
            msg = "Input field must be 2D"
            raise ValueError(msg)

        self._input_field_fft = field_fft

    def set_scattering_potential(self, potential: Array) -> None:
        """Set the 3D scattering potential.

        Parameters
        ----------
        potential : `jax.Array`
            3D scattering potential array

        Raises
        ------
        ValueError
            If potential is not 3D
        """
        if len(potential.shape) != _NDIM_3D:
            msg = "Scattering potential must be 3D"
            raise ValueError(msg)

        self._scattering_potential = potential

    def _initialize_field_arrays(self, save_all_layers: bool) -> tuple[Array | None, Array | None]:
        """Initialize field arrays based on save_all_layers option.

        Returns
        -------
        `tuple`[`jax.Array` | None, `jax.Array` | None]
            (fields_fft, current_field_fft) where one will be None based on save_all_layers
        """
        if save_all_layers:
            fields_fft = jnp.zeros(
                (self.params.num_layers, *self.params.xy_shape),
                dtype=jnp.complex64,
            )
            fields_fft = fields_fft.at[0].set(self._input_field_fft)
            return fields_fft, None
        return None, self._input_field_fft

    @staticmethod
    def _get_previous_field(
        layer_idx: int, save_all_layers: bool, fields_fft: Array | None, current_field_fft: Array | None
    ) -> Array:
        """Get the previous layer field based on storage strategy.

        Returns
        -------
        `jax.Array`
            Previous layer field

        Raises
        ------
        RuntimeError
            If field arrays are in unexpected state
        """
        if save_all_layers:
            if fields_fft is None:
                msg = "fields_fft should not be None when save_all_layers=True"
                raise RuntimeError(msg)
            return fields_fft[layer_idx - 1]
        if current_field_fft is None:
            msg = "current_field_fft should not be None when save_all_layers=False"
            raise RuntimeError(msg)
        return current_field_fft

    @staticmethod
    def _update_field_arrays(
        layer_idx: int,
        next_field_fft: Array,
        save_all_layers: bool,
        fields_fft: Array | None,
        current_field_fft: Array | None,
    ) -> tuple[Array | None, Array | None]:
        """Update field arrays with new field.

        Returns
        -------
        `tuple`[`jax.Array` | None, `jax.Array` | None]
            Updated (fields_fft, current_field_fft) tuple

        Raises
        ------
        RuntimeError
            If field arrays are in unexpected state
        """
        if save_all_layers:
            if fields_fft is None:
                msg = "fields_fft should not be None when save_all_layers=True"
                raise RuntimeError(msg)
            updated_fields_fft = fields_fft.at[layer_idx].set(next_field_fft)
            return updated_fields_fft, current_field_fft
        return fields_fft, next_field_fft

    def simulate_forward_scattering(self, save_all_layers: bool = False) -> Array:
        """Simulate multi-layer Born forward scattering.

        Parameters
        ----------
        save_all_layers : `bool`, default=False
            If True, return fields from all layers. If False, return only final field.

        Returns
        -------
        `jax.Array`
            Scattered electromagnetic fields. Shape is (num_layers, H, W) if
            save_all_layers=True, otherwise (H, W).

        Raises
        ------
        ValueError
            If input field or scattering potential is not set
        RuntimeError
            If internal field arrays are in unexpected state
        """
        if self._input_field_fft is None:
            msg = "Input field not set. Call set_input_field_fft() first."
            raise ValueError(msg)

        if self._scattering_potential is None:
            msg = "Scattering potential not set. Call set_scattering_potential() first."
            raise ValueError(msg)

        # Initialize field arrays
        fields_fft, current_field_fft = self._initialize_field_arrays(save_all_layers)

        # Pre-compute propagation kernels
        propagator_dz = get_propagator(self.params, self.params.dz_m)
        green_func_dz = get_green_func(self.params, self.params.dz_m)

        # Multi-layer forward scattering loop
        for layer_idx in tqdm(range(1, self.params.num_layers), desc="MLB forward scattering"):
            # Get previous field
            prev_field_fft = MLBForward._get_previous_field(layer_idx, save_all_layers, fields_fft, current_field_fft)

            # Unscattered component (free propagation)
            unscattered_fft = propagator_dz * prev_field_fft

            # Scattered component
            # 1. Convert to spatial domain
            prev_field_spatial = jnp.fft.ifft2(jnp.fft.ifftshift(prev_field_fft))

            # 2. Apply scattering potential
            scattered_source = prev_field_spatial * self._scattering_potential[layer_idx - 1] * self.params.dz_m

            # 3. Convert back to frequency domain and apply Green's function
            scattered_source_fft = jnp.fft.fftshift(jnp.fft.fft2(scattered_source))
            scattered_fft = green_func_dz * scattered_source_fft

            # Total field at next layer
            next_field_fft = unscattered_fft + scattered_fft

            # Update field arrays
            fields_fft, current_field_fft = MLBForward._update_field_arrays(
                layer_idx, next_field_fft, save_all_layers, fields_fft, current_field_fft
            )

        # Return appropriate result
        if save_all_layers:
            if fields_fft is None:
                msg = "fields_fft should not be None when save_all_layers=True"
                raise RuntimeError(msg)
            return fields_fft

        if current_field_fft is None:
            msg = "current_field_fft should not be None when save_all_layers=False"
            raise RuntimeError(msg)
        return current_field_fft

    def get_observation_field(
        self,
        field_fft: Array | None = None,
        propagate_to_focus: bool = True,
    ) -> Array:
        """Get the field at the observation plane (detector).

        Parameters
        ----------
        field_fft : `jax.Array`, optional
            Field in frequency domain. If None, uses result from forward simulation.
        propagate_to_focus : `bool`, default=True
            Whether to propagate to the focal plane

        Returns
        -------
        `jax.Array`
            2D electromagnetic field at observation plane
        """
        if field_fft is None:
            field_fft = self.simulate_forward_scattering()

        # Convert to spatial domain
        field_spatial = jnp.fft.ifft2(jnp.fft.ifftshift(field_fft))

        if propagate_to_focus:
            # Propagate to focus (typically back to the middle of the sample)
            focus_distance_px = -self.params.num_layers // 2
            field_spatial = propagate(field_spatial, focus_distance_px, self.params)

        # Apply collection aperture
        field_fft_collected = jnp.fft.fftshift(jnp.fft.fft2(field_spatial))
        aperture = get_aperture(self.params)
        field_fft_collected *= aperture

        return jnp.fft.ifft2(jnp.fft.ifftshift(field_fft_collected))
