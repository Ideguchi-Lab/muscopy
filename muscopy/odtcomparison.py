"""ODT Comparison

Here we compare a digitally developed ground truth sphere of various sizes
and reconstruct it using ODT. We then compare the result to the original to see
for what sizes and what refractive index ODT is most effective
"""

import typing
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from muscopy.cfg import ArrayPrecision
from muscopy.dh import get_spectrum, print_all_parameters
from muscopy.odt import ODTConfig, ODTParameters, odt
from tqdm import tqdm
from muscopy.odt_with_mlb_simulation import generate_sphere_potential, MLBParameters, compute_odt


#Define axes
n_values = jnp.linspace(1.33, 1.5, 20)
r_values = jnp.linspace(0.5, 10, 20)

n_grid, r_grid = jnp.meshgrid(n_values, r_values, indexing='ij')

print_all_parameters

#Compute Error
def compute_error(n: float, r: float, n_recon) -> float:
    """Finds the error between the Ground Truth image and ODT Reconstruction
    given an n value and an r value

    Parameters
    ----------
    n : `float`
        refractive index of sphere
    r : `float`
        raidus of sphere

    Returns
    -------
    ODT error
    """
    nr_pairs = [(n, r) for n in n_values for r in r_values]
    for n, r in nr_pairs:
        mlb_params = MLBParameters(1.0, 1.33, 20, [123, 62], 123, 20, 20)
        gt_image = generate_sphere_potential(mlb_params, r, n).astype(float)
        recon_volume = n_recon.astype(float)
    
        if gt_image.shape != recon_volume.shape:
            raise ValueError(f"Shape mismatch: gt_image has shape {gt_image.shape}, and recon_volume has shape {recon_volume.shape}")
    
        z_idx = gt_image.shape[2] // 2
        gt_slice = gt_image[:, :, z_idx]
        recon_slice = recon_volume[:, :, z_idx]
        error = jnp.mean(jnp.abs(gt_slice - recon_slice)**2)
    return float(error)

error_grid = jnp.array([[compute_error(n, r, compute_odt(n, r)) for n in n_values] for r in r_values])

print("error_grid.shape:", error_grid.shape)
print("error_grid:", error_grid)

#Plot
plt.figure(figsize=(20,20))
plt.imshow(error_grid,
           extent=[n_values.min(), n_values.max(), r_values.min(), r_values.max()],
           origin="lower",
           aspect="auto",
           cmap="viridis",
           vmin=error_grid.min(),
           vmax=error_grid.max())
plt.xlabel("Refractive Index")
plt.ylabel("Sphere Radius (um)")
plt.colorbar(label="Error (|GT - ODT|^2)")
plt.title("Error Heatmap")
plt.tight_layout()
plt.show()