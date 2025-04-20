"""Demonstration of Quantitative Phase Imaging (QPI) reconstruction."""

# %%
# import modules

import matplotlib.pyplot as plt

from muscopy.backend_manager import BackendManager
from muscopy.qpi import QPIParameters, make_disk, print_qpi_all_parameters, qpi

# config
SHOW_IMAGE = True

# %%
# set QPI parameters

params = QPIParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)

# show QPI parameters
print_qpi_all_parameters(params, show_properties=True)

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

ref_sample_array = backend.ones_like(sample_array)
ref_sample_array /= backend.sum(backend.abs(ref_sample_array) ** 2) ** 0.5
ref_hologram = backend.abs(ref_sample_array + ref_array) ** 2

# %%
# Extract phase of scattering wave with QPI

phase_image = qpi(backend, hologram, ref_hologram, params, [off_axis_center])[0]
if bmg.backend == "cupy":
    phase_image = backend.asnumpy(phase_image)

if SHOW_IMAGE:
    plt.imshow(phase_image, cmap="gray")
    plt.colorbar()
    plt.show()
# %%
