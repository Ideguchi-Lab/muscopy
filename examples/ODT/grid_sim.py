# %%
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from ilabvis import CursorVisualizer, SlicingVisualizer

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
    extract3Dto2D_minimum,
    generate_3d_gaussian,
    generate_3D_slope,
    generate_3D_sphere,
    generate_plate,
    generate_test_data,
)

from muscopy.aperture_synthesis import Synthesizer as QPISynthesizer
from muscopy.dir_parser import numpy_parser
from muscopy.odt import (
    ODTParameters,
    ODTSynthesizer,
    calc_normalized_L2error,
    calc_refractive_index_square,
    discard_higher_kz,
    zeropad_higher_kz,
)

EDGE_SIZE = 0

sizes = [5, 10, 25]
n_indeices = [1.35, 1.4, 1.5]
rot_angles = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
NAs = [(0.8, 0.6), (1.2, 1.0), (1.3, 1.28)]
Hermite_list = [True, False]

for size, n_index, rot_angle, NA, hermite in product(sizes, n_indeices, rot_angles, NAs, Hermite_list):

    # %%
    NA_i = NA[1]
    step_angle = 360 / rot_angle
    params = ODTParameters(
        532e-9,
        NA[0],
        (1400, 1400),
        (700, 700),
        (3.45 * 1e-6) * 3 / 200 / 5,
        # (3.45 * 1e-6) * 3 / 200 / 2,
        (612, 623),
        1.33,
        # 1.48,
        NA_i,
    )
    params.calc_params()
    params.print_all_parameters()

    # shape for z-compressed 3D map (for iterative ODT)
    shape_3d = (
        params.aperturesize * 2 + 1,
        params.aperturesize * 2 + 1,
        # params.fz_extent * 2 + 1,
        params.aperturesize * 2 + 1,
    )

    sample_index = n_index  # PMMA=1.49, water=1.33
    radius = size
    # hermite = True
    # hermite = False

    print("real beads size:", 2 * radius * params.imgpx_unit * 1e6, "um")

    # make answer 3D refractive map
    sphere = generate_3D_sphere(
        # shape=params.aperturesize * 2 + 1,
        shape=shape_3d,
        radius=radius,
        center=(params.aperturesize, params.aperturesize, params.aperturesize),
    )

    depth = 30
    plate = generate_plate(
        shape_3d,
        (params.aperturesize, params.aperturesize, params.aperturesize),
        30,
        depth,
    )

    gauss = generate_3d_gaussian(
        shape_3d,
        (params.aperturesize, params.aperturesize, params.aperturesize),
        radius,
    )

    PT_radius = 5
    PT_dn = (
        0.1  # in the real set up, order may be 1e-2. but, current set up, 1e-2 modulation may be difficult to detect
    )
    PT_gauss = generate_3d_gaussian(
        shape_3d,
        (params.aperturesize + 5, params.aperturesize + 5, params.aperturesize),
        PT_radius,
    )

    # slope = generate_3D_slope(shape_3d, "x")

    # slope = slope / slope.shape[0]  # normalize

    r_sample = sphere
    # r_sample = gauss
    # r_sample = slope
    # r_sample = plate

    uniform_background = discard_higher_kz(
        xp.ones(r_sample.shape) * params.n_sol, params.aperturesize - params.fz_extent
    )

    # create target and reference refractive index map
    PT = False
    ref_rindex = xp.ones(r_sample.shape) * params.n_sol
    rindex_original = ref_rindex + r_sample * (sample_index - params.n_sol)

    # comment out below when MIP-ODT
    # PT = True
    # ref_rindex = xp.ones(r_sample.shape) * params.n_sol + r_sample * (
    #     sample_index - params.n_sol
    # )
    # rindex_original = ref_rindex + PT_gauss * (PT_dn * (sample_index - params.n_sol))

    rindex = discard_higher_kz(rindex_original, params.aperturesize - params.fz_extent)
    ref_rindex = discard_higher_kz(ref_rindex, params.aperturesize - params.fz_extent)
    # then calculate scattering potential based on refractive index
    scatter_potential = -1 * ((params.k_per_pixel * params.fi_mag)) ** 2 * (rindex**2 / uniform_background**2 - 1)

    norm_scatter_potential = scatter_potential * params.imgpx_unit * params.imgpx_unit_z**0.5
    norm_scatter_fft = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(norm_scatter_potential, axes=(2)), norm="backward"))
    scatter_fft = norm_scatter_fft / params.k_per_pixel ** (3 / 2)
    # scatter_fft[0:EDGE_SIZE, :, :] = 0
    # scatter_fft[:, :, 0:EDGE_SIZE] = 0
    # scatter_fft[:, 0:EDGE_SIZE, :] = 0
    ref_scatter_potential = -((params.k_per_pixel * params.fi_mag) ** 2) * (ref_rindex**2 / uniform_background**2 - 1)

    norm_ref_scatter_potential = ref_scatter_potential * params.imgpx_unit * params.imgpx_unit_z**0.5
    norm_ref_scatter_fft = xp.fft.fftshift(
        xp.fft.fftn(xp.fft.ifftshift(norm_ref_scatter_potential, axes=(2)), norm="backward")
    )
    ref_scatter_fft = norm_ref_scatter_fft / params.k_per_pixel ** (3 / 2)
    # ref_scatter_fft[0:EDGE_SIZE, :, :] = 0
    # ref_scatter_fft[:, 0:EDGE_SIZE, :] = 0
    # ref_scatter_fft[:, :, 0:EDGE_SIZE] = 0

    approx_field_fft = scatter_fft / 2j
    ref_approx_field_fft = ref_scatter_fft / 2j

    # # %%
    # # show the given refractive index
    # print("original refractive index")
    # rindex_to_show = xp.asnumpy(rindex_original)
    # slice_visualizer = SlicingVisualizer(rindex_to_show)
    # slice_visualizer.run()

    # # %%
    # print("truncated refractive index")
    # rindex_to_show = xp.asnumpy(xp.abs(rindex))
    # slice_visualizer = SlicingVisualizer(rindex_to_show)
    # slice_visualizer.run()

    # # %%
    # # directly retrieve 3D fft spectrum
    # ret_index = xp.zeros(
    #     rindex.shape,
    #     dtype=xp.int32,
    # )

    # for angle in range(0, 360, int(step_angle)):
    #     oblique_shift = (
    #         int(params.fi_lateral_mag * np.cos(np.deg2rad(angle))),
    #         int(params.fi_lateral_mag * np.sin(np.deg2rad(angle))),
    #     )
    #     print(oblique_shift)
    #     map = extract3Dto2D_minimum(
    #         ref_rindex, params, oblique_shift, return_index_map=True
    #     )
    #     ret_index += map

    # ret_index = ret_index > 0

    # if _cp:
    #     ret_index = xp.asnumpy(ret_index)

    # slice_visualizer = SlicingVisualizer(ret_index)
    # slice_visualizer.run()

    # # %%
    # # retrieve 3D fft spectrum
    # if _cp:
    #     ret_index = xp.array(ret_index)
    # ret_fft = scatter_fft * ret_index

    # norm_ret_fft = ret_fft * params.k_per_pixel ** (3 / 2)

    # # ret_array = (params.freq_per_pixel) ** 3 * xp.fft.fftshift(
    # #     xp.fft.ifftn(xp.fft.ifftshift(ret_fft)), axes=(2)
    # # )
    # norm_ret_array = xp.fft.fftshift(
    #     xp.fft.ifftn(xp.fft.ifftshift(norm_ret_fft), norm="backward"), axes=(2)
    # )
    # ret_array = norm_ret_array / (params.imgpx_unit * params.imgpx_unit_z**0.5)
    # # ret_ref = xp.real(params.n_sol * (1 - calc_refractive_index_square(ret_array, params)) ** 0.5)
    # # if _cp:
    # #     ret_ref = xp.asnumpy(ret_ref)

    # # print("refractive index with true scatter potential and true index")
    # # slice_visualizer = SlicingVisualizer(ret_ref)
    # # slice_visualizer.run()

    # # %%
    # # zero pad higher kz
    # ret_array_padded = zeropad_higher_kz(ret_array, params.aperturesize - params.fz_extent)
    # ret_ref_padded = xp.real(calc_refractive_index_square(ret_array_padded, params) ** 0.5)
    # if PT:
    #     ret_ref_padded = ret_ref_padded - params.n_sol
    # if _cp:
    #     ret_ref_padded = xp.asnumpy(ret_ref_padded)
    # print("zeropad refractive index with true scatter potential and true index")
    # slice_visualizer = SlicingVisualizer(ret_ref_padded)
    # slice_visualizer.run()

    # %%
    # generate hologram
    step_angle = step_angle
    NA_illumi = NA_i
    # approx = "Born"
    approx = "Rytov"

    generate_test_data(
        approx_field_fft,
        params,
        step_angle,
        NA_illumi,
        "odt_test_data/sample",
        approx=approx,
        if_save=True,
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

    # # %%
    # # cursor visualizer
    # data2d = np.load("odt_test_data/sample/180.npy")
    # print(f"shape: {data2d.shape}")
    # # spectrum = np.fft.fftshift(np.fft.fft2(data2d))
    # spectrum = data2d
    # cursor_visualizer = CursorVisualizer(np.log(np.abs(spectrum) + 1))
    # cursor_visualizer.run()

    # %%
    # execute synthetic aperture
    load_fft = True
    test_data = numpy_parser("odt_test_data/sample")
    ref_data = numpy_parser("odt_test_data/ref")
    # load_fft = False
    # test_data = numpy_parser("../data/aperture_sample_beads")
    # ref_data = numpy_parser("../data/aperture_ref_beads")
    qpi_synthesizer = QPISynthesizer()
    qpi_synthesizer.set_parameters(params)
    # qpi_synthesizer.set_data(test_data)
    qpi_synthesizer.set_data(test_data, ref_data)
    qpi_synthesized, qpi_fft = qpi_synthesizer.synthesize(save_multiangle=True, load_fft=load_fft)

    if _cp:
        qpi_synthesized = xp.asnumpy(qpi_synthesized)
        qpi_fft = xp.asnumpy(qpi_fft)

    # plot synthesized QPI with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(qpi_synthesized)
    # plt.colorbar()
    # plt.show()
    # print("max phase: ", np.max(qpi_synthesized))
    # print("min phase: ", np.min(qpi_synthesized))
    # print(
    #     "estimated phase:",
    #     2 * np.pi * 2 * radius
    #     # * depth
    #     * params.imgpx_unit * (sample_index - params.n_sol) / params.wav,
    # )
    # plt.savefig("synthesized_qpi.png")

    sample_phase = np.max(qpi_synthesized)
    estimated_phase = 2 * np.pi * 2 * radius * params.imgpx_unit * (sample_index - params.n_sol) / params.wav

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.log(np.abs(qpi_fft)))
    ax.scatter(qpi_fft.shape[0] // 2, qpi_fft.shape[1] // 2, s=10, c="red")
    # plt.savefig("synthesized_qpi_fft.png", dpi=300)

    # cursor_visualizer = CursorVisualizer(qpi_synthesized)
    # cursor_visualizer.run()

    # %%
    odt_synthesizer = ODTSynthesizer()
    odt_synthesizer.set_parameters(params)
    odt_synthesizer.set_data(test_data, ref_data)
    synthesized_array, odt_fft, occupancy = odt_synthesizer.ODT_synthesize(
        approx=approx, hermite=hermite, load_fft=load_fft
    )
    r_index_map = xp.real(calc_refractive_index_square(synthesized_array, params) ** 0.5) - params.n_sol

    synthesized_array_pad = zeropad_higher_kz(synthesized_array, params.aperturesize - params.fz_extent)
    r_index_map_pad = xp.real(calc_refractive_index_square(synthesized_array_pad, params) ** 0.5)

    normalized_error = calc_normalized_L2error(r_index_map_pad, rindex_original)
    # print(f"normalized error: {normalized_error}")

    max_ref_index = xp.max(r_index_map)
    ref_diff = (sample_index - params.n_sol) - max_ref_index
    # print(f"refractive index difference: {max_ref_index - (sample_index - params.n_sol)}")

    if PT:
        r_index_map = r_index_map - params.n_sol
        r_index_map_pad = r_index_map_pad - params.n_sol

    print("====================================")
    print(f"{occupancy=}, {sample_phase=}, {estimated_phase=}, {normalized_error=}, {ref_diff=}")

    fn = "result.txt"
    with open(fn, "a") as f:
        f.write(
            f"\n{size}, {n_index}, {rot_angle}, {NA[0]}, {NA[1]}, {hermite}, {occupancy}, {sample_phase}, {estimated_phase}, {normalized_error}, {ref_diff}"
        )

    # imag_scatter_potential = xp.imag(
    #     calc_refractive_index_square(synthesized_array, params) ** 0.5
    # )

    # # %%
    # print("imag part of scatter potential")
    # print(f"max: {xp.max(imag_scatter_potential)}")
    # slice_visualizer = SlicingVisualizer(xp.asnumpy(imag_scatter_potential))
    # slice_visualizer.run()

    # # %%
    # scatter_pot_abs = xp.asnumpy(xp.abs(ret_array))
    # ret_scatter_pot_abs = xp.asnumpy(xp.abs(synthesized_array))

    # ratio = xp.sum(xp.abs(ret_array) ** 2) / xp.sum(xp.abs(synthesized_array) ** 2)
    # print(ratio)

    # print("scatter potential")
    # slice_visualizer = SlicingVisualizer(scatter_pot_abs)
    # slice_visualizer.run()

    # print("ret scatter potential")
    # slice_visualizer = SlicingVisualizer(ret_scatter_pot_abs)
    # slice_visualizer.run()

    # %%

    # if _cp:
    #     odt_synthesized = xp.asnumpy(r_index_map)
    #     odt_synthesized_pad = xp.asnumpy(r_index_map_pad)
    #     odt_fft = xp.asnumpy(xp.log(xp.abs(odt_fft) + 1))
    # else:
    #     odt_synthesized = r_index_map
    #     odt_synthesized_pad = xp.real(r_index_map_pad)
    #     odt_fft = xp.asnumpy(xp.log(xp.abs(odt_fft) + 1))

    # # %%
    # # plot
    # print("synthesized ODT")
    # slice_visualizer = SlicingVisualizer(odt_synthesized_pad)
    # slice_visualizer.run()

    # # %%
    # # plot
    # print("fft index map")
    # fft_exist = np.array(odt_fft != 0)
    # t_slice_visualizer = SlicingVisualizer(fft_exist)
    # slice_visualizer.run()

    # # %%
    # # print(np.testing.assert_almost_equal(xp.asnumpy(ret_index).astype(int), fft_exist))
    # print(np.sum(np.abs(xp.asnumpy(ret_index) - fft_exist)))

    # diff_index = xp.asnumpy(ret_index) - fft_exist
    # slice_visualizer = SlicingVisualizer(diff_index)
    # slice_visualizer.run()

    # # %%
    # # check the difference between the true scatter potential and the synthesized scatter potential
    # ret_fft_new = scatter_fft * xp.array(fft_exist)

    # norm_ret_fft_new = ret_fft_new * params.k_per_pixel ** (3 / 2)
    # norm_ret_array_new = xp.fft.fftshift(
    #     xp.fft.ifftn(xp.fft.ifftshift(norm_ret_fft_new), norm="backward"), axes=(2)
    # )
    # ret_array_new = norm_ret_array_new / (params.imgpx_unit * params.imgpx_unit_z**0.5)
    # ret_ref_new = xp.real(calc_refractive_index_square(ret_array_new, params) ** 0.5)
    # ret_array_new = xp.asnumpy(ret_ref_new)

    # # print(np.testing.assert_almost_equal(ret_ref, odt_synthesized))
    # print(np.sum(np.abs(ret_ref - odt_synthesized)))
    # print(np.sum(np.abs(ret_array_new - odt_synthesized)))
    # diff_ref = ret_array_new - odt_synthesized
    # # slice_visualizer = SlicingVisualizer(diff_ref)
    # # slice_visualizer.run()

    # to_show = xp.asnumpy(ret_array_new)

    # print("true scatter potential with wrong fft index")
    # slice_visualizer = SlicingVisualizer(to_show)
    # slice_visualizer.run()

    # # %%
    # # iterative ODT
    # odt_synthesizer = ODTSynthesizer()
    # odt_synthesizer.set_parameters(params)
    # # odt_synthesizer.set_data(test_data, ref_data)
    # odt_synthesizer.set_data(test_data, ref_data)
    # synthesized_array, odt_fft = odt_synthesizer.iterative_ODT(
    #     approx=approx, epsilon=1e-6, max_N=10000, hermite=hermite, load_fft=load_fft
    # )

    # synthesized_array = xp.real(
    #     calc_refractive_index_square(synthesized_array, params) ** 0.5
    # )

    # if _cp:
    #     odt_synthesized = xp.asnumpy(synthesized_array)
    #     odt_fft = xp.asnumpy(xp.log(xp.abs(odt_fft) + 1))
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

    # # # # %%
