# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from ilabvis import SlicingVisualizer

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


# %%
params = ODTParameters(
    532e-9, 1.2, (1400, 1400), (700, 700), (3.45 * 1e-6) * 3 / 200 / 5, (623, 612), 1.33
)
params.calc_params()
params.print_all_parameters()

# make answer 3D refractive map
sphere = generate_3D_sphere(
    shape=params.aperturesize * 2 + 1,
    radius=params.aperturesize / 5.0,
    center=(params.aperturesize, params.aperturesize, params.aperturesize),
)
rindex = sphere * 0.01 * 100
amp_map = xp.ones(sphere.shape) + sphere * 0.1
complex_field = amp_map * xp.exp(1j * rindex)

# convert to fft space
array_3d_fft = xp.fft.fftshift(xp.fft.fftn(complex_field))

# %%
# generate hologram
step_angle = 360 / 10
NA_illumi = 1.0

generate_test_data(array_3d_fft, params, step_angle, NA_illumi)
print("generated test data!")

# # %%
# # debug for test data
# t_data = np.load("odt_test_data/000.npy")
# t_data_fft = np.fft.fftshift(np.fft.fftn(t_data))

# plt.imshow(np.log(np.abs(t_data_fft)))
# plt.scatter(*params.offaxis_center, s=1)
# # plt.scatter(params.offaxis_center[1], params.offaxis_center[0], s=1)

# # %%
# # visualize 3D refractive index
# if _cp:
#     rindex_to_show = xp.asnumpy(rindex)
# else:
#     rindex_to_show = rindex
# slice_visualizer = SlicingVisualizer(rindex_to_show)
# slice_visualizer.run()

# # %%
# # visualize 3D fft spectrum
# if _cp:
#     array_3d_fft_to_show = xp.asnumpy(xp.log(xp.abs(array_3d_fft)))
# else:
#     array_3d_fft_to_show = xp.log(xp.abs(array_3d_fft))
# slice_visualizer = SlicingVisualizer(array_3d_fft_to_show)
# slice_visualizer.run()

# %%
# execute synthetic aperture
test_data = numpy_parser("odt_test_data")
# test_data = numpy_parser("../data/aperture_sample_beads")
# ref_data = numpy_parser("../data/aperture_ref_beads")
qpi_synthesizer = QPISynthesizer()
qpi_synthesizer.set_parameters(params)
qpi_synthesizer.set_data(test_data)
# qpi_synthesizer.set_data(test_data, ref_data)
qpi_synthesized, qpi_fft = qpi_synthesizer.synthesize(save_multiangle=True)

if _cp:
    qpi_synthesized = xp.asnumpy(qpi_synthesized)
    qpi_fft = xp.asnumpy(qpi_fft)

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
# odt_synthesizer.set_data(test_data, ref_data)
odt_synthesizer.set_data(test_data)
synthesized_array, odt_fft = odt_synthesizer.ODT_synthesize("Rytov")

if _cp:
    odt_synthesized = xp.asnumpy(xp.real(synthesized_array))
    odt_fft = xp.asnumpy(np.log(np.abs(odt_fft) + 1))
else:
    odt_synthesized = np.real(synthesized_array)
    odt_fft = np.log(np.abs(odt_fft) + 1)

# %%
# plot
slice_visualizer = SlicingVisualizer(odt_synthesized)
slice_visualizer.run()

# %%
# plot
slice_visualizer = SlicingVisualizer(odt_fft)
slice_visualizer.run()

# %%
# iterative ODT
odt_synthesizer = ODTSynthesizer()
odt_synthesizer.set_parameters(params)
# odt_synthesizer.set_data(test_data, ref_data)
odt_synthesizer.set_data(test_data)
synthesized_array, odt_fft = odt_synthesizer.iterative_ODT(epsilon=1e-6, max_N=1000)

if _cp:
    odt_synthesized = xp.asnumpy(xp.real(synthesized_array))
    odt_fft = xp.asnumpy(np.log(np.abs(odt_fft) + 1))
else:
    odt_synthesized = np.real(synthesized_array)
    odt_fft = np.log(np.abs(odt_fft) + 1)
# %%
# plot
slice_visualizer = SlicingVisualizer(odt_synthesized)
slice_visualizer.run()

# %%
# plot
slice_visualizer = SlicingVisualizer(odt_fft)
slice_visualizer.run()
