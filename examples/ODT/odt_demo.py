# %%
import sys
import numpy as np
import matplotlib.pyplot as plt
from ilabvis import SlicingVisualizer, CursorVisualizer

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
from src.odt import ODTParameters, ODTSynthesizer, calc_refractive_index_square


# %%
params = ODTParameters(
    532e-9, 1.2, (1401, 1401), (700, 700), (3.45 * 1e-6) * 3 / 200 / 5, (623, 612), 1.33
)
params.calc_params()
params.print_all_parameters()

sample_index = 1.35

# make answer 3D refractive map
sphere = generate_3D_sphere(
    shape=params.aperturesize * 2 + 1,
    radius=10,
    center=(params.aperturesize, params.aperturesize, params.aperturesize),
)

# create target and reference refractive index map
ref_rindex = xp.ones(sphere.shape) * params.n_sol
rindex = ref_rindex + sphere * (sample_index - params.n_sol)
# then calculate scattering potential based on refractive index
scatter_potential = -1 * params.ki_mag**2 * (rindex**2 / ref_rindex**2 - 1)
scatter_fft = xp.fft.fftshift(xp.fft.fftn(scatter_potential))
ref_scatter_potential = -params.ki_mag**2 * (ref_rindex**2 / ref_rindex**2 - 1)
ref_scatter_fft = xp.fft.fftshift(xp.fft.fftn(ref_scatter_potential))

xx, yy, zz = xp.meshgrid(
    xp.arange(params.aperturesize * 2 + 1),
    xp.arange(params.aperturesize * 2 + 1),
    xp.arange(params.aperturesize * 2 + 1),
    indexing="ij",
)
kz_array = zz - params.aperturesize + params.ki_mag
approx_field_fft = scatter_fft / kz_array / 2j
ref_approx_field_fft = ref_scatter_fft / kz_array / 2j

# %%
# generate hologram
step_angle = 360 / 10
NA_illumi = 1.0
# approx = "Born"
approx = "Rytov"

generate_test_data(
    approx_field_fft,
    params,
    step_angle,
    NA_illumi,
    "odt_test_data/sample",
    approx=approx,
)
generate_test_data(
    ref_approx_field_fft,
    params,
    step_angle,
    NA_illumi,
    "odt_test_data/ref",
    approx=approx,
)
print("generated test data!")

# %%
# cursor visualizer
data2d = np.load("odt_test_data/sample/000.npy")
print(f"shape: {data2d.shape}")
spectrum = np.fft.fftshift(np.fft.fft2(data2d))
cursor_visualizer = CursorVisualizer(np.log(np.abs(spectrum)))
cursor_visualizer.run()

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
# array_3d_fft = xp.fft.fftn(rindex)
# if _cp:
#     array_3d_fft_to_show = xp.asnumpy(xp.log(xp.abs(array_3d_fft)))
# else:
#     array_3d_fft_to_show = xp.log(xp.abs(array_3d_fft))
# slice_visualizer = SlicingVisualizer(array_3d_fft_to_show)
# slice_visualizer.run()

# %%
# execute synthetic aperture
test_data = numpy_parser("odt_test_data/sample")
ref_data = numpy_parser("odt_test_data/ref")
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
print("max phase: ", np.max(qpi_synthesized))
print("min phase: ", np.min(qpi_synthesized))
plt.savefig("synthesized_qpi.png")

plt.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.log(np.abs(qpi_fft)))
ax.scatter(qpi_fft.shape[0] // 2, qpi_fft.shape[1] // 2, s=10, c="red")
plt.savefig("synthesized_qpi_fft.png", dpi=300)


# %%
test_data = numpy_parser("odt_test_data/sample")
ref_data = numpy_parser("odt_test_data/ref")
odt_synthesizer = ODTSynthesizer()
odt_synthesizer.set_parameters(params)
# odt_synthesizer.set_data(test_data, ref_data)
odt_synthesizer.set_data(test_data, ref_data)
synthesized_array, odt_fft = odt_synthesizer.ODT_synthesize(
    approx=approx, hermite=False
)

synthesized_array = (
    xp.abs(calc_refractive_index_square(synthesized_array, params)) ** 0.5
)

if _cp:
    odt_synthesized = xp.asnumpy(synthesized_array)
    odt_fft = xp.asnumpy(xp.log(xp.abs(odt_fft) + 1))
else:
    odt_synthesized = xp.real(synthesized_array)
    odt_fft = xp.asnumpy(xp.log(xp.abs(odt_fft) + 1))

# %%
# plot
slice_visualizer = SlicingVisualizer(odt_synthesized)
slice_visualizer.run()

# %%
# plot
slice_visualizer = SlicingVisualizer(odt_fft)
slice_visualizer.run()

# # %%
# # iterative ODT
# odt_synthesizer = ODTSynthesizer()
# odt_synthesizer.set_parameters(params)
# # odt_synthesizer.set_data(test_data, ref_data)
# odt_synthesizer.set_data(test_data, ref_data)
# synthesized_array, odt_fft = odt_synthesizer.iterative_ODT(
#     approx=approx, epsilon=1e-6, max_N=1000
# )

# synthesized_array = (
#     xp.abs(calc_refractive_index_square(synthesized_array, params)) ** 0.5
# )

# if _cp:
#     odt_synthesized = xp.asnumpy(synthesized_array)
#     odt_fft = xp.asnumpy(np.log(np.abs(odt_fft) + 1))
# else:
#     odt_synthesized = synthesized_array
#     odt_fft = np.log(np.abs(odt_fft) + 1)
# # %%
# # plot
# slice_visualizer = SlicingVisualizer(odt_synthesized)
# slice_visualizer.run()

# # %%
# # plot
# slice_visualizer = SlicingVisualizer(odt_fft)
# slice_visualizer.run()

# %%
