"""
Demonstration of Quantitative Phase Imaging (QPI) reconstruction
================================================================

This example demonstrates how to use the `muscopy.qpi` function to reconstruct the phase of a
sample from an off-axis hologram of a disk-shaped phase object.
"""

# %%
# import modules

import jax
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, print_all_parameters
from muscopy.example_holograms import off_axis_holograms
from muscopy.qpi import correct_phase_offset, qpi

# config
SHOW_IMAGE = True

print(jax.default_backend())

# %%
# set Microscopy parameters

params = MuParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)

# show QPI parameters
print_all_parameters(params, show_properties=True)

# %%
# Generate synthetic measurements
off_axis_center = (100, 100)
hologram, ref_hologram = off_axis_holograms(params, off_axis_center)

# %%
# Extract phase of scattering wave with QPI

offset_regions = [
    ((5, 10), (5, 10)),
    ((params.aperturesize_px - 10, params.aperturesize_px - 5), (5, 10)),
    ((5, 10), (params.aperturesize_px - 10, params.aperturesize_px - 5)),
    (
        (params.aperturesize_px - 10, params.aperturesize_px - 5),
        (params.aperturesize_px - 10, params.aperturesize_px - 5),
    ),
]

phase_image = qpi(hologram, ref_hologram, params, off_axis_center)

# correct the phase offset by using the corner regions
phase_image = correct_phase_offset(phase_image, offset_regions)

if SHOW_IMAGE:
    phase_image_to_show = jax.device_get(phase_image)
    plt.imshow(phase_image_to_show)
    plt.colorbar()
    plt.show()
# %%
