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


#Define axes
n_values = jnp.linspace(1.33, 1.5, 20)
r_values = jnp.linspace(0.5, 3, 20)

n_grid, r_grid = jnp.meshgrid(n_values, r_values, indexing='ij')


#Compute Error
def compute_error(n: float, r: float) -> float:
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
    n_recon = compute_odt(n, r)

    mlb_params = MLBParameters(r, n, 20, [123, 62], 123, 20, 20)
    gt_image = generate_sphere_potential(mlb_params, r, n).astype(float)
    recon_volume = n_recon.astype(float)
    
    if gt_image.shape != recon_volume.shape:
        raise ValueError(f"Shape mismatch: gt_image has shape {gt_image.shape}, and recon_volume has shape {recon_volume.shape}")
    
    z_idx = gt_image.shape[2] // 2
    gt_slice = gt_image[:, :, z_idx]
    recon_slice = recon_volume[:, :, z_idx]
    error = jnp.mean(jnp.abs(gt_slice - recon_slice)**2)
    return (error)

def generate_error_map(n_values, r_values):
    errors = jnp.zeros((len(r_values), len(n_values)))

    for i, r in enumerate(r_values):
        for j, n in enumerate(n_values):
            errors = errors.at[i, j].set(compute_error(n, r))
    return errors

errors = generate_error_map(n_values, r_values)

#Plot
plt.figure(figsize=(5,5))
sns.heatmap(errors,
           annot=True,
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
           
plt.xlabel("Refractive Index")
plt.ylabel("Sphere Radius (um)")
plt.title("Error Heatmap")
plt.tight_layout()
plt.show()