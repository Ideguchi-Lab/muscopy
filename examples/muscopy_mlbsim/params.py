"""Simulation parameters for the MLBSim module.

This module provides:

- `SimParameters`: A dataclass that encapsulates the parameters required for MLB simulation
- Compatibility wrapper for the new MLBParameters class
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from muscopy_mlbsim.mlb import MLBParameters


@dataclasses.dataclass
class SimParameters:
    """Simulation parameters for Multi-layer Born simulation.

    This is a dataclass-based interface that provides compatibility with the
    MLBParameters class while maintaining a simple dataclass structure.

    Attributes
    ----------
    numerical_aperture : `float`
        Numerical aperture of the objective lens
    wavelength_m : `float`
        Wavelength of the light in meters
    xy_size : `int`
        Size of the simulation grid in the x and y dimensions (assumes square grid)
    num_layers : `int`
        Number of layers in the simulation
    dxy_m : `float`
        Lateral grid spacing in meters
    dz_m : `float`
        Axial grid spacing in meters
    n_medium : `float`, default=1.33
        Refractive index of the surrounding medium (default: water)
    """

    numerical_aperture: float
    wavelength_m: float
    xy_size: int
    num_layers: int
    dxy_m: float
    dz_m: float
    n_medium: float = 1.33

    def to_mlb_parameters(self) -> MLBParameters:
        """Convert to MLBParameters instance.

        Returns
        -------
        `MLBParameters`
            Equivalent MLBParameters instance
        """
        from muscopy_mlbsim.mlb import MLBParameters  # noqa: PLC0415

        return MLBParameters(
            wavelength_m=self.wavelength_m,
            numerical_aperture=self.numerical_aperture,
            n_medium=self.n_medium,
            xy_shape=(self.xy_size, self.xy_size),
            num_layers=self.num_layers,
            dxy_m=self.dxy_m,
            dz_m=self.dz_m,
        )

    @classmethod
    def from_mlb_parameters(cls, params: MLBParameters) -> SimParameters:
        """Create from MLBParameters instance.

        Parameters
        ----------
        params : `MLBParameters`
            Source MLBParameters instance

        Returns
        -------
        `SimParameters`
            New SimParameters instance

        Raises
        ------
        ValueError
            If xy_shape is not square
        """
        if params.xy_shape[0] != params.xy_shape[1]:
            msg = "SimParameters only supports square grids"
            raise ValueError(msg)

        return cls(
            numerical_aperture=params.numerical_aperture,
            wavelength_m=params.wavelength_m,
            xy_size=params.xy_shape[0],
            num_layers=params.num_layers,
            dxy_m=params.dxy_m,
            dz_m=params.dz_m,
            n_medium=params.n_medium,
        )
