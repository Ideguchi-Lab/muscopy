"""
Demonstration of Quantitative Phase Imaging (QPI) reconstruction
================================================================

This example demonstrates how to use the `muscopy.qpi` function to reconstruct the phase of a
sample from a hologram. The example uses a simple off-axis holography setup with a disk-shaped
"""

# %%
# import modules

import math

import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, make_disk, print_all_parameters
from muscopy.qpi import correct_phase_offset, qpi

# config
SHOW_IMAGE = True

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
# create a hologram
off_axis_center = (100, 100)

radius = 30
sample_disk = make_disk(params.img_center, radius, params.img_size_px)
sample_array = jnp.exp(2j * math.pi / 3 * sample_disk)
low_pass = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
sample_array = jnp.fft.ifft2(jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fft2(sample_array)) * low_pass))
sample_array /= jnp.sum(jnp.abs(sample_array) ** 2) ** 0.5

xx, yy = jnp.meshgrid(
    jnp.arange(params.img_size_px),
    jnp.arange(params.img_size_px),
    indexing="ij",
)
ref_array = jnp.exp(
    -2j
    * jnp.pi
    * (
        xx * (off_axis_center[0] - params.img_center[0]) / (params.img_size_px)
        + yy * (off_axis_center[1] - params.img_center[1]) / (params.img_size_px)
    )
)
ref_array /= jnp.sum(jnp.abs(ref_array) ** 2) ** 0.5

hologram = jnp.abs(sample_array + ref_array) ** 2

ref_sample_array = jnp.ones_like(sample_array)
ref_sample_array /= jnp.sum(jnp.abs(ref_sample_array) ** 2) ** 0.5
ref_hologram = jnp.abs(ref_sample_array + ref_array) ** 2

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
    plt.imshow(phase_image, cmap="gray")
    plt.colorbar()
    plt.show()
# %%
