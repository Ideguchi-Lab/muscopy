"""
Phase Unwrapping Demo
=================================

Demonstrate the unwrap phase functionality of the muscopy library.
"""

# %%

import jax
import matplotlib.pyplot as plt
import numpy as np
from muscopy_example_data.holograms import gaussian_phase_object, off_axis_holograms
from numpy.typing import NDArray
from skimage.restoration import unwrap_phase as skimage_unwrap_phase

from muscopy.dh import MuParameters, print_all_parameters
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
# Generate synthetic measurements
off_axis_center = (100, 100)
hologram, ref_hologram = off_axis_holograms(params, off_axis_center, gaussian_phase_object(params))

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

unwrapped_skimage: NDArray[np.float64] = skimage_unwrap_phase(phase_image_np)  # type: ignore[no-untyped-call]

if SHOW_IMAGE:
    plt.imshow(unwrapped_skimage)
    plt.colorbar()
    plt.title("Unwrapped phase with skimage")
    plt.show()
