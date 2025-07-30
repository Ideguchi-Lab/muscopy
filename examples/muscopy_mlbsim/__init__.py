"""Multi-layer Born simulation plugin for muscopy library.

This package provides high-quality implementations of Multi-layer Born (MLB)
forward scattering simulation for optical diffraction tomography applications.

Key components:
- `MLBParameters`: Core parameter class for MLB simulations
- `MLBForward`: Forward scattering simulation engine
- `HologramGenerator`: Digital hologram generation utilities
- `SimParameters`: Dataclass-based parameter interface for compatibility
"""

from muscopy_mlbsim.hologram_generator import HologramGenerator
from muscopy_mlbsim.mlb import (
    MLBForward,
    MLBParameters,
    get_aperture,
    get_fx_fy_phaser,
    get_fz_phaser,
    get_green_func,
    get_oblique_wave,
    get_oblique_wave_fft,
    get_propagator,
    get_refractive_index,
    get_scatter_potential,
    propagate,
)
from muscopy_mlbsim.params import SimParameters

__all__ = [
    "HologramGenerator",
    "MLBForward",
    "MLBParameters",
    "SimParameters",
    "get_aperture",
    "get_fx_fy_phaser",
    "get_fz_phaser",
    "get_green_func",
    "get_oblique_wave",
    "get_oblique_wave_fft",
    "get_propagator",
    "get_refractive_index",
    "get_scatter_potential",
    "propagate",
]
