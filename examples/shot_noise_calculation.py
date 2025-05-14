"""
Shot noise calculation from a hologram
====================================================

Demonstration of the shot noise calculation from a hologram
"""

# %%

import matplotlib.pyplot as plt

from muscopy.backend_manager import BackendManager
from muscopy.dh import MuParameters, make_disk, print_all_parameters
from muscopy.phase_noise import calc_phase_noise, calc_visibility

# %%
# config
SHOW_IMAGE = True

# Image sensor parameters
full_well_capacity = int(10e3)  # full well capacity (e-)
bit_depth = 12  # bit depth of the camera
sensor_noise = 10  # sensor noise (e-)

# %%
# generate a hologram

params = MuParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)
print_all_parameters(params, show_properties=True)

# %%
# specify the backend
bmg = BackendManager()
bmg.use_numpy()

print(f"{bmg.backend} backend is used.")
backend = bmg.get_backend()

# %%
# create a hologram
off_axis_center = (100, 100)

radius = 30
sample_disk = make_disk(backend, params.img_center, radius, params.img_size_px)
sample_array = backend.exp(2j * backend.pi / 3 * sample_disk)
low_pass = make_disk(backend, (params.img_center), params.aperturesize_px // 2, params.img_size_px)
sample_array = backend.fft.ifft2(backend.fft.ifftshift(backend.fft.fftshift(backend.fft.fft2(sample_array)) * low_pass))
sample_array /= backend.sum(backend.abs(sample_array) ** 2) ** 0.5

xx, yy = backend.meshgrid(
    backend.arange(params.img_size_px),
    backend.arange(params.img_size_px),
    indexing="ij",
)
ref_array = backend.exp(
    -2j
    * backend.pi
    * (
        xx * (off_axis_center[0] - params.img_center[0]) / (params.img_size_px)
        + yy * (off_axis_center[1] - params.img_center[1]) / (params.img_size_px)
    )
)
ref_array /= backend.sum(backend.abs(ref_array) ** 2) ** 0.5

hologram = backend.abs(sample_array + ref_array) ** 2
# normalize
hologram /= backend.max(hologram)  # normalize to 1
hologram *= 2**bit_depth

# %%
# calculate visibility and phase noise
visibility = calc_visibility(backend, hologram, params, off_axis_center)
phase_noise = calc_phase_noise(backend, hologram, params, off_axis_center, full_well_capacity, bit_depth, sensor_noise)


# %%
if SHOW_IMAGE:
    plt.imshow(visibility)
    plt.title("Visibility")
    plt.colorbar()
    plt.show()
    plt.imshow(phase_noise)
    plt.title("Phase noise")
    plt.colorbar()
    plt.show()

# %%
