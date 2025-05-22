"""
Phase Unwrapping Demo
=================================

Demonstrate the unwrap phase functionality of the muscopy library.
"""

# %%

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import unwrap_phase as skimage_unwrap_phase

from muscopy.dh import MuParameters, make_disk, print_all_parameters
from muscopy.qpi import qpi
from muscopy.qpi_utils import unwrap_phase as mus_unwrap_phase

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

# show Microscopy parameters
print_all_parameters(params, show_properties=True)


# %%
# create a hologram
off_axis_center = (100, 100)

x_radius = 30
y_radius = 100
xx, yy = jnp.meshgrid(
    jnp.arange(-params.img_size_px // 2, params.img_size_px // 2),
    jnp.arange(-params.img_size_px // 2, params.img_size_px // 2),
    indexing="ij",
)
gaussian = 2 * jnp.exp(-((xx) ** 2 / (2 * (x_radius / 2) ** 2) + (yy) ** 2 / (2 * (y_radius / 2) ** 2)))
sample_array = jnp.exp(2j * jnp.pi * gaussian)
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

phase_image = qpi(hologram, ref_hologram, params, off_axis_center)

if SHOW_IMAGE:
    phase_image_to_show = jax.device_get(phase_image)
    plt.imshow(phase_image_to_show)
    plt.colorbar()
    plt.show()

# %%
# Unwrap phase with muscopy
unwrapped_mus = mus_unwrap_phase(phase_image)

if SHOW_IMAGE:
    unwrapped_mus_to_show = jax.device_get(unwrapped_mus)
    plt.imshow(unwrapped_mus_to_show)
    plt.colorbar()
    plt.title("Unwrapped phase with muscopy")
    plt.show()

# %%
# Reference unwrapped phase with skimage

phase_image_np = np.asarray(phase_image)

unwrapped_skimage = skimage_unwrap_phase(phase_image_np)

if SHOW_IMAGE:
    plt.imshow(unwrapped_skimage)
    plt.colorbar()
    plt.title("Unwrapped phase with skimage")
    plt.show()
