"""ODT Comparison

Here we compare a digitally developed ground truth sphere of various sizes
and reconstruct it using ODT. We then compare the result to the original to see
for what sizes and what refractive index ODT is most effective
"""

import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt

from muscopy.odt import calc_refractive_index, ODTParameters
from muscopy.odt_with_mlb_simulation import compute_odt

# Define axes
n_values = jnp.linspace(1.33, 1.5, 5)
r_values = jnp.linspace(0.5, 3, 5)

n_grid, r_grid = jnp.meshgrid(n_values, r_values, indexing="ij")

odt_params = ODTParameters(
    na=1.1,
    wavelength_m=532e-9,
    img_size_px=512,  # Smaller for faster computation
    px_size_m=3.45e-6 * 3 / 180 / 2,
    n_sol=1.33,
    na_illumination=1.0,
)


# Compute Error
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
    error = jnp.mean(jnp.abs(gt_slice - recon_slice) ** 2)
    return float(error)


error_grid = jnp.zeros((len(r_values), len(n_values)))
for i, r in enumerate(r_values):
    for j, n in enumerate(n_values):
        n_recon, gt_potential = compute_odt(n, r)
        gt_r_index = calc_refractive_index(gt_potential, odt_params) - odt_params.n_sol

        error_grid = error_grid.at[i, j].set(compute_error(n_recon, gt_r_index))

print("error_grid.shape:", error_grid.shape)
print("error_grid:", error_grid)

# Plot
plt.figure(figsize=(20, 20))
plt.imshow(
    error_grid,
    extent=[n_values.min(), n_values.max(), r_values.min(), r_values.max()],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    vmin=error_grid.min(),
    vmax=error_grid.max(),
)
plt.xlabel("Refractive Index")
plt.ylabel("Sphere Radius (um)")
plt.colorbar(label="Error (|GT - ODT|^2)")
plt.title("Error Heatmap")
plt.tight_layout()
plt.show()
