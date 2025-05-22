"""
Demonstration of multiplexed QPI
================================

This example demonstrates how to use the `muscopy.dh.demultiplex_cp_arrays` function to demultiplex
complex fields obtained from multiplexed QPI. The example uses a simple multiplexing matrix and
"""

# %%
# import modules
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, demultiplex_cp_arrays, make_disk, offaxis_dh, print_all_parameters

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

# show parameters
print_all_parameters(params, show_properties=True)

# %%
# multiplex matrix
multiplex_matrix = jnp.asarray(
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
sample_disk1 = make_disk(params.img_center, radius1, params.img_size_px)
sample_disk2 = make_disk(pos2, radius2, params.img_size_px)
sample_array1 = jnp.exp(2j * jnp.pi / 3 * sample_disk1)
sample_array2 = jnp.exp(2j * jnp.pi / 5 * sample_disk2)
low_pass = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
sample_array1 = jnp.fft.ifft2(jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fft2(sample_array1)) * low_pass))
sample_array2 = jnp.fft.ifft2(jnp.fft.ifftshift(jnp.fft.fftshift(jnp.fft.fft2(sample_array2)) * low_pass))
sample_array1 /= jnp.sum(jnp.abs(sample_array1) ** 2) ** 0.5
sample_array2 /= jnp.sum(jnp.abs(sample_array2) ** 2) ** 0.5


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

hologram1 = jnp.abs(multiplex_matrix[0, 0] * sample_array1 + multiplex_matrix[0, 1] * sample_array2 + ref_array) ** 2
hologram2 = jnp.abs(multiplex_matrix[1, 0] * sample_array1 + multiplex_matrix[1, 1] * sample_array2 + ref_array) ** 2

ref_sample_array = jnp.ones_like(sample_array1)
ref_sample_array /= jnp.sum(jnp.abs(ref_sample_array) ** 2) ** 0.5
ref_hologram = jnp.abs(ref_sample_array + ref_array) ** 2

# %%
# extract complex fields

cp_field1 = offaxis_dh(hologram1, ref_hologram, params, off_axis_center)
cp_field2 = offaxis_dh(hologram2, ref_hologram, params, off_axis_center)

if SHOW_IMAGE:
    angle1 = jax.device_get(jnp.angle(cp_field1))
    angle2 = jax.device_get(jnp.angle(cp_field2))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(angle1, cmap="gray")
    ax[0].set_title("Complex field 1")
    ax[1].imshow(angle2, cmap="gray")
    ax[1].set_title("Complex field 2")
    plt.show()

cp_fields = [cp_field1, cp_field2]

# %%
# calculate the demultiplexing matrix
# transfer the multiplexing matrix to CPU due to cuSolver internal error
cpu = jax.devices("cpu")[0]
multiplex_matrix = jax.device_put(multiplex_matrix, device=cpu)
demultiplexing_matrix = jnp.linalg.inv(multiplex_matrix)
demultiplexing_matrix = jax.device_put(demultiplexing_matrix)


# %%
# demultiplex complex fields

demultiplexed_cp_fields = demultiplex_cp_arrays(cp_fields, demultiplexing_matrix)

qpi1 = jnp.angle(demultiplexed_cp_fields[0])
qpi2 = jnp.angle(demultiplexed_cp_fields[1])

# %%
# show QPI images

if SHOW_IMAGE:
    qpi1_to_show = jax.device_get(qpi1)
    qpi2_to_show = jax.device_get(qpi2)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(qpi1_to_show, cmap="gray")
    ax[0].set_title("QPI 1")
    ax[1].imshow(qpi2_to_show, cmap="gray")
    ax[1].set_title("QPI 2")
    plt.show()
