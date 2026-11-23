"""
Demonstration of multiplexed QPI
================================

This example demonstrates how to use the `muscopy.dh.demultiplex_cp_arrays` function to demultiplex
complex fields obtained from multiplexed QPI. A two-channel multiplexing matrix mixes two phase objects.
"""

# %%
# import modules
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from muscopy_example_data.holograms import disk_phase_object, off_axis_holograms

from muscopy.dh import MuParameters, demultiplex_cp_arrays, offaxis_dh, print_all_parameters

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
# Generate multiplexed measurements
off_axis_center = (100, 100)
sample_fields = jnp.stack(
    (
        disk_phase_object(params),
        disk_phase_object(params, radius_px=10, phase_step_rad=2 * jnp.pi / 5, center=(100, 200)),
    )
)
holograms, ref_hologram = off_axis_holograms(params, off_axis_center, sample_fields, multiplex_matrix=multiplex_matrix)

# %%
# extract complex fields

cp_field1 = offaxis_dh(holograms[0], ref_hologram, params, off_axis_center)
cp_field2 = offaxis_dh(holograms[1], ref_hologram, params, off_axis_center)

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
