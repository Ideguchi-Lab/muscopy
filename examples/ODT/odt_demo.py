# %%
import os
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

from generate_hologram_from3Dmap import generate_3D_sphere, generate_test_data

from src.dir_parser import numpy_parser
from src.aperture_synthesis import Synthesizer as QPISynthesizer
from src.odt import ODTParameters, ODTSynthesizer

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_matrix(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Get the coordinates of non-zero points
    xpos, ypos, zpos = np.where(matrix > 0)

    # Loop through the points and plot each one
    for x, y, z in zip(xpos, ypos, zpos):
        ax.scatter(x, y, z, c="b", marker="o")  # Change color and marker as needed

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


# Example usage:
# Assuming upper_semi_sphere_matrix is a 3D numpy array representing the upper-semisphere
# plot_3d_matrix(upper_semi_sphere_matrix)

# %%
# params = params = ODTParameters(
#     532e-9, 1.2, (400, 400), (200, 200), (3.45 * 1e-6) * 3 / 200 / 4, (50, 100), 1.33
# )
params = ODTParameters(
    532e-9, 1.2, (1400, 1400), (700, 700), (3.45 * 1e-6) * 3 / 200 / 5, (623, 612), 1.33
)
params.calc_params()
params.print_all_parameters()

# # make answer 3D refractive map
# rindex = (
#     generate_3D_sphere(
#         shape=params.aperturesize * 2,
#         radius=params.aperturesize / 5.0,
#         center=(params.aperturesize, params.aperturesize, params.aperturesize),
#     )
#     * 2
# )
# complex_field = xp.ones(rindex.shape, dtype=xp.complex128) * xp.exp(1j * rindex)

# # convert to fft space
# array_3d_fft = xp.fft.fftshift(xp.fft.fftn(complex_field))

# # %%
# # generate hologram
# step_angle = 360 / 10
# NA_illumi = 1

# generate_test_data(array_3d_fft, params, step_angle, NA_illumi)
# print("generated test data!")

# %%
# execute synthetic aperture
# test_data = numpy_parser("odt_test_data")
test_data = numpy_parser("../data/aperture_sample_beads")
ref_data = numpy_parser("../data/aperture_ref_beads")
qpi_synthesizer = QPISynthesizer()
qpi_synthesizer.set_parameters(params)
qpi_synthesizer.set_data(test_data, ref_data)
qpi_synthesized, qpi_fft = qpi_synthesizer.synthesize(save_multiangle=True)

if _cp:
    qpi_synthesized = xp.asnumpy(qpi_synthesized)

# plot synthesized QPI with color bar
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(qpi_synthesized)
# plt.colorbar()
# plt.show()

plt.savefig("synthesized_qpi.png")

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log(np.abs(qpi_fft)))
ax.scatter(qpi_fft.shape[0] // 2, qpi_fft.shape[1] // 2, s=10, c="red")
plt.savefig("synthesized_qpi_fft.png", dpi=300)


# %%
odt_synthesizer = ODTSynthesizer()
odt_synthesizer.set_parameters(params)
odt_synthesizer.set_data(test_data, ref_data)
odt_synthesized, odt_fft = odt_synthesizer.ODT_synthesize()

# plot 3d matrix using matplotlib
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(np.log(np.abs(odt_fft)))
# ax.scatter(odt_fft.shape[0] // 2, odt_fft.shape[1] // 2, s=10, c="red")
# plt.savefig("synthesized_odt_fft.png", dpi=300)
plot_3d_matrix(np.log(np.abs(odt_fft)))

# %%
