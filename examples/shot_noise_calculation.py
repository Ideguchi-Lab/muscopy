"""
Shot noise calculation from a hologram
====================================================

Demonstration of the shot noise calculation from a hologram
"""

# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, print_all_parameters
from muscopy.example_holograms import off_axis_holograms
from muscopy.phase_noise import calc_phase_noise, calc_visibility

# %%
# config
SHOW_IMAGE = True

# Image sensor parameters
full_well_capacity = int(10e3)  # full well capacity (e-)
bit_depth = 12  # bit depth of the camera
sensor_noise = 10  # sensor noise (e-)

# %%
# Set microscopy parameters

params = MuParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)
print_all_parameters(params, show_properties=True)

# %%
# Generate synthetic measurements
off_axis_center = (100, 100)
hologram, _ = off_axis_holograms(params, off_axis_center)
hologram = hologram / jnp.max(hologram) * 2**bit_depth

# %%
# calculate visibility and phase noise
visibility = calc_visibility(hologram, params, off_axis_center)
phase_noise = calc_phase_noise(hologram, params, off_axis_center, full_well_capacity, bit_depth, sensor_noise)


# %%
if SHOW_IMAGE:
    visibility_to_show = jax.device_get(visibility)
    phase_noise_to_show = jax.device_get(phase_noise)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    img1 = ax[0].imshow(visibility_to_show)
    ax[0].set_title("Visibility")
    img2 = ax[1].imshow(phase_noise_to_show)
    ax[1].set_title("Phase noise")
    # colorbar
    fig.colorbar(img1, ax=ax[0])
    fig.colorbar(img2, ax=ax[1])
    plt.show()

# %%
