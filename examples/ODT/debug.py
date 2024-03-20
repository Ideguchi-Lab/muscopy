# %%
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

backend = "numpy" if not _cp else "cupy"
print(f"backend: {backend}")

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from generate_hologram_from3Dmap import (
    generate_3D_sphere,
    project_onto_2D_space,
    extract3Dto2D,
)

from src.odt import ODTParameters

# %%
# params = ODTParameters(
#     532e-9, 1.2, (1400, 1400), (700, 700), (3.45 * 1e-6) * 3 / 200 / 5, (623, 612), 1.33
# )
params = ODTParameters(
    532e-9, 1.2, (400, 400), (200, 200), (3.45 * 1e-6) * 3 / 200 / 1, (100, 100), 1.33
)
# params = ODTParameters(
#     532e-9, 1.2, (1400, 1400), (700, 700), (3.45 * 1e-6) * 3 / 200, (297, 277), 1.33
# )
params.calc_params()
params.print_all_parameters()
# generate 3D sphere
sphere = generate_3D_sphere(
    shape=params.aperturesize * 2,
    radius=params.aperturesize / 10.0,
    center=(params.aperturesize, params.aperturesize, params.aperturesize),
)
complex_field = xp.ones(sphere.shape, dtype=xp.complex128) * xp.exp(1j * sphere * 1.5)
# plot sphere in matplotlib
# ax = plt.figure().add_subplot(projection="3d")
# ax.voxels(sphere, linewidth=0.5)
# ax.set(xlabel="r", ylabel="g", zlabel="b")
# ax.set_aspect("equal")

# plt.show()

# %%
# convert to fft space

sphere_fft = xp.fft.fftshift(xp.fft.fftn(sphere))
# if _cp:
#     sphere_fft_to_show = xp.asnumpy(sphere_fft)
# else:
#     sphere_fft_to_show = sphere_fft
# # plot sphere in matplotlib
# ax = plt.figure().add_subplot(projection="3d")
# ax.voxels((np.abs(sphere_fft_to_show)), linewidth=0.5)
# ax.set(xlabel="r", ylabel="g", zlabel="b")
# ax.set_aspect("equal")

# # %%
# fft_2d = project_onto_2D_space(sphere_fft)
# # plot with matplotlib
# fig, ax = plt.subplots()
# im = ax.imshow(np.abs(fft_2d))
# pp = fig.colorbar(im, ax=ax)
# pp.set_label("amplitude")
# plt.show()

# %%
# extract 2D fft array
extracted_2d_fft = extract3Dto2D(sphere_fft, params, 0, 0.3)
# plot with matplotlib
fig, ax = plt.subplots()
im = ax.imshow(np.log(np.abs(extracted_2d_fft)))
pp = fig.colorbar(im, ax=ax)
pp.set_label("amplitude")
plt.show()


# %%
# reconstruct 2D image
sphere_recon = xp.fft.ifft2(xp.fft.ifftshift(extracted_2d_fft))
if _cp:
    sphere_recon_to_show = xp.asnumpy(xp.abs(sphere_recon))
else:
    sphere_recon_to_show = xp.abs(sphere_recon)
# plot with matplotlib
fig, ax = plt.subplots()
im = ax.imshow(sphere_recon_to_show)
pp = fig.colorbar(im, ax=ax)
pp.set_label("phase")
plt.show()

# %%
