"""Demonstration of multiplexed QPI."""

# %%
# import modules
import matplotlib.pyplot as plt

from muscopy.backend_manager import BackendManager
from muscopy.dh import demultiplex_cp_arrays
from muscopy.qpi import QPIParameters, make_disk, offaxis_dh, print_all_parameters

# config
SHOW_IMAGE = True

# %%
# set DH parameters

params = QPIParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)

# show parameters
print_all_parameters(params, show_properties=True)

# %%
# specify the backend
bmg = BackendManager()
bmg.use_numpy()

print(f"{bmg.backend} backend is used.")
backend = bmg.get_backend()

# %%
# multiplex matrix
multiplex_matrix = backend.asarray(
    [
        [1 / 2, 1 / 2],
        [1j / 2, -1j / 2],
    ]
)

# %%
# create a multiplexed holograms
off_axis_center = (100, 100)

radius1 = 30
radius2 = 10
pos2 = (100, 200)
sample_disk1 = make_disk(backend, params.img_center, radius1, params.img_size_px)
sample_disk2 = make_disk(backend, pos2, radius2, params.img_size_px)
sample_array1 = backend.exp(2j * backend.pi / 3 * sample_disk1)
sample_array2 = backend.exp(2j * backend.pi / 5 * sample_disk2)
low_pass = make_disk(backend, (params.img_center), params.aperturesize_px // 2, params.img_size_px)
sample_array1 = backend.fft.ifft2(
    backend.fft.ifftshift(backend.fft.fftshift(backend.fft.fft2(sample_array1)) * low_pass)
)
sample_array2 = backend.fft.ifft2(
    backend.fft.ifftshift(backend.fft.fftshift(backend.fft.fft2(sample_array2)) * low_pass)
)
sample_array1 /= backend.sum(backend.abs(sample_array1) ** 2) ** 0.5
sample_array2 /= backend.sum(backend.abs(sample_array2) ** 2) ** 0.5


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

hologram1 = (
    backend.abs(multiplex_matrix[0, 0] * sample_array1 + multiplex_matrix[0, 1] * sample_array2 + ref_array) ** 2
)
hologram2 = (
    backend.abs(multiplex_matrix[1, 0] * sample_array1 + multiplex_matrix[1, 1] * sample_array2 + ref_array) ** 2
)

ref_sample_array = backend.ones_like(sample_array1)
ref_sample_array /= backend.sum(backend.abs(ref_sample_array) ** 2) ** 0.5
ref_hologram = backend.abs(ref_sample_array + ref_array) ** 2

# %%
# extract complex fields

cp_field1 = offaxis_dh(backend, hologram1, ref_hologram, params, [off_axis_center])[0]
cp_field2 = offaxis_dh(backend, hologram2, ref_hologram, params, [off_axis_center])[0]

cp_fields = [cp_field1, cp_field2]

# %%
# calculate the demultiplexing matrix

demultiplexing_matrix = backend.linalg.inv(multiplex_matrix)


# %%
# demultiplex complex fields

demultiplexed_cp_fields = demultiplex_cp_arrays(bmg, cp_fields, demultiplexing_matrix)

qpi1 = backend.angle(demultiplexed_cp_fields[0])
qpi2 = backend.angle(demultiplexed_cp_fields[1])

# %%
# show QPI images

if bmg.backend == "cupy":
    qpi1 = backend.asnumpy(qpi1)
    qpi2 = backend.asnumpy(qpi2)

if SHOW_IMAGE:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(qpi1, cmap="gray")
    ax[0].set_title("QPI 1")
    ax[1].imshow(qpi2, cmap="gray")
    ax[1].set_title("QPI 2")
    plt.show()
