"""ODT Comparison

Here we compare a digitally developed ground truth sphere of various sizes
and reconstruct it using ODT. We then compare the result to the original to see
for what sizes and what refractive index ODT is most effective
"""

import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from muscopy.odt_with_mlb_simulation import generate_sphere_potential, MLBParameters, compute_odt
from muscopy.odt import calc_refractive_index, ODTParameters
"""
NOTE: To run this you have to edit muscopy/odt_with_mlb_simulation.py
Line 484: change ' -> Array' to ' -> 'tuple[Array, Array]'
Line 528: change 'return n_reconstructed' to 'return n_reconstructed, scattering_potential'
"""
#Define axes
n_values = jnp.linspace(0.01, 0.17, 20)
r_values = jnp.linspace(0.5, 3, 20)

n_grid, r_grid = jnp.meshgrid(n_values, r_values, indexing='ij')
odt_params = ODTParameters(
    na=1.1,
    wavelength_m=532e-9,
    img_size_px=512,
    px_size_m=3.45e-6 * 3 /180 / 2,
    n_sol=1.33,
    na_illumination=1.0
)
#Compute Error
def compute_error(n_recon: Array, gt_r_index: Array) -> float:
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

    gt_image = jnp.real(gt_r_index)
    recon_volume = jnp.real(n_recon)
    
    z_idx = gt_image.shape[2] // 2
    gt_slice = gt_image[z_idx, :, :]
    recon_slice = recon_volume[:, :, z_idx]
    error = jnp.mean(jnp.abs(gt_slice - recon_slice)**2)
    return float(error)


error_grid = jnp.zeros((len(r_values), len(n_values)))

for i, r in enumerate(r_values):
    for j, n in enumerate(n_values):
        n_recon, gt_potential = compute_odt(n, r)
        
        gt_r_index = calc_refractive_index(gt_potential, odt_params) - odt_params.n_sol

        error_grid = error_grid.at[i, j].set(compute_error(n_recon, gt_r_index))

#Plot
plt.figure(figsize=(8,6))
ax = sns.heatmap(error_grid,
           annot=False,
           cmap="viridis", 
           linewidths=0.1,
           linecolor="black",
           cbar_kws={"label":"ODT Error Map"},
           xticklabels=False,
           yticklabels=False)
plt.xticks(
    ticks=range(len(n_values)),
    labels=[f"{n:.3f}" for n in n_values],
    rotation=45
)
plt.yticks(
    ticks=range(len(r_values)),
    labels=[f"{r:.2f}" for r in r_values],
    rotation=0
)
ax.invert_yaxis()
           
plt.xlabel("Refractive Index")
plt.ylabel("Sphere Radius (um)")
plt.title("Error Heatmap")
plt.tight_layout()
plt.show()