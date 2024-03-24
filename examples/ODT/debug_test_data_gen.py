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

from generate_hologram_from3Dmap import *

from muscopy.odt import ODTParameters, ODTSynthesizer, calc_refractive_index_square

# %%
params = ODTParameters(532e-9, 1.2, (1400, 1400), (700, 700), (3.45 * 1e-6) * 3 / 200 / 5, (623, 612), 1.33)
params.calc_params()
params.print_all_parameters()

sample_index = 1.35
# %%
sphere = generate_3D_sphere(
    shape=params.aperturesize * 2 + 1,
    radius=params.aperturesize / 5.0,
    center=(params.aperturesize, params.aperturesize, params.aperturesize),
)
ref_rindex = xp.ones(sphere.shape) * params.n_sol
rindex = ref_rindex + sphere * (sample_index - params.n_sol)
scatter_potential = -1 * params.ki_mag**2 * (rindex**2 / ref_rindex**2 - 1)
scatter_fft = xp.fft.fftshift(xp.fft.fftn(scatter_potential))
ref_scatter_potential = -params.ki_mag**2 * (ref_rindex**2 / ref_rindex**2 - 1)
print("ref_scatter potential sum: ", xp.sum(ref_scatter_potential))
ref_scatter_fft = xp.fft.fftshift(xp.fft.fftn(ref_scatter_potential))

xx, yy, zz = xp.meshgrid(
    xp.arange(params.aperturesize * 2 + 1),
    xp.arange(params.aperturesize * 2 + 1),
    xp.arange(params.aperturesize * 2 + 1),
    indexing="xy",
)
kz_array = zz - params.aperturesize + params.ki_mag
approx_field_fft = scatter_fft / kz_array / 2j
ref_approx_field_fft = ref_scatter_fft / kz_array / 2j

# %%

E_initial = xp.ones((2 * params.aperturesize + 1, 2 * params.aperturesize + 1), dtype=xp.complex128)

# NA_illumi = 1.0
NA_illumi = 0
# i = 10
i = 1
# illumi_angle_step = 30
illumi_angle_step = 360
array_3d_fft = approx_field_fft
ki = round(NA_illumi / params.wav / params.freq_per_pixel)  # + 1 why +1?
oblique_shift = (
    int(ki * xp.cos(illumi_angle_step * i / 360 * 2 * xp.pi)),
    int(ki * xp.sin(illumi_angle_step * i / 360 * 2 * xp.pi)),
)
print(f"{oblique_shift=}")
test_data_fft_cropped = extract3Dto2D_minimum(array_3d_fft, params, oblique_shift=oblique_shift)

# %%
fft_to_show = xp.asnumpy((xp.abs(test_data_fft_cropped)))
cursorvis = CursorVisualizer(fft_to_show)
cursorvis.run()

# %%

fft_extent = xp.zeros(
    (2 * params.aperturesize + 1, 2 * params.aperturesize + 1),
    dtype=xp.complex128,
)
fft_extent[
    params.aperturesize
    - oblique_shift[0]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[0]
    + params.aperturesize // 2
    + 1,
    params.aperturesize
    - oblique_shift[1]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[1]
    + params.aperturesize // 2
    + 1,
] = test_data_fft_cropped

to_show = xp.asnumpy(xp.log(xp.abs(fft_extent)))
plt.imshow(to_show)
plt.scatter(params.aperturesize, params.aperturesize, s=100, c="red")

# %%
test_data_extent = xp.fft.ifft2(xp.fft.ifftshift(fft_extent))
test_data_extent[:1, :] = 0
test_data_extent[:, :1] = 0

# E_test = E_initial + test_data_extent
E_test = E_initial * xp.exp(test_data_extent / E_initial)

plt.imshow(xp.asnumpy(xp.abs(E_test)))
# %%
plt.imshow(xp.asnumpy(xp.angle(E_test)))

# %%
E_test_fft = xp.fft.fftshift(xp.fft.fft2(E_test))
low_pass = make_disk(
    (
        params.aperturesize - oblique_shift[0],
        params.aperturesize - oblique_shift[1],
    ),
    params.aperturesize // 2,
    E_initial.shape,
)

plt.imshow(xp.asnumpy(xp.log(xp.abs(low_pass))))

# %%
E_test_fft = E_test_fft * low_pass
test_data_fft = xp.zeros(params.img_shape, dtype=xp.complex128)

# %%
to_show = E_test_fft[
    params.aperturesize
    - oblique_shift[1]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[1]
    + params.aperturesize // 2
    + 1,
    params.aperturesize
    - oblique_shift[0]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[0]
    + params.aperturesize // 2
    + 1,
]
plt.imshow(xp.asnumpy(xp.log(xp.abs(to_show))))

# %%

test_data_fft[
    params.offaxis_center[0] - params.aperturesize // 2 : params.offaxis_center[0] + params.aperturesize // 2 + 1,
    params.offaxis_center[1] - params.aperturesize // 2 : params.offaxis_center[1] + params.aperturesize // 2 + 1,
] = E_test_fft[
    params.aperturesize
    - oblique_shift[1]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[1]
    + params.aperturesize // 2
    + 1,
    params.aperturesize
    - oblique_shift[0]
    - params.aperturesize // 2 : params.aperturesize
    - oblique_shift[0]
    + params.aperturesize // 2
    + 1,
]

to_show = xp.asnumpy(xp.log(xp.abs(test_data_fft)))
plt.imshow(to_show)
plt.scatter(params.offaxis_center[1], params.offaxis_center[0], s=1, c="red")
# %%
