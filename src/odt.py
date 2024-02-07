from __future__ import annotations

import os
import sys

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from src.aperture_synthesis import Synthesizer, preprocess_for_synthesis, EDGE_SIZE
from src.qpi import QPIParameters, make_disk, qpi

EDGE_SIZE = 0  # for avoiding edge artifact in ifft


def find_max_args(array: NDArray):
    """Find max x, y coordinates and value of the given array.
    Given matrix data type must be real-value type.

    Args:
        array (NDArray): input array

    Returns:
        tuple:  x, y coordinates and value of the max value
    """
    max_value = xp.max(array)
    max_idx = xp.unravel_index(np.argmax(array), array.shape)
    max_x = max_idx[0]
    max_y = max_idx[1]

    return max_x, max_y, max_value


class ODTParameters(QPIParameters):
    def __init__(
        self,
        wavelength,
        NA,
        img_shape,
        img_center,
        pixelsize,
        offaxis_center,
        n_sol,
        NA_illumi=None,
    ):
        super().__init__(
            wavelength, NA, img_shape, img_center, pixelsize, offaxis_center
        )
        self.n_sol = n_sol
        self.NA_illumi = NA_illumi

    def calc_params(self):
        super().calc_params()
        self.ki_mag = self.n_sol / self.wav / self.freq_per_pixel

        self.k_unit = 2 * np.pi * self.freq_per_pixel
        # self.k_unit = self.freq_per_pixel
        self.imgpx_unit = (
            self.pixelsize * self.img_shape[0] / (2 * self.aperturesize + 1)
        )
        # self.imgpx_unit = 1 / self.k_unit

        if self.NA_illumi is not None:
            self.ki_lateral_mag = self.ki_mag * self.NA_illumi / self.NA

    def print_all_parameters(self):
        super().print_all_parameters()
        print(f"{self.n_sol=}")
        print(f"{self.ki_mag=}")
        print(f"{self.ki_lateral_mag=}")
        print(f"{self.k_unit=}")
        print(f"{self.imgpx_unit=}")


def reconstruct_E(
    array: NDArray,
    ref_array: NDArray,
    params: ODTParameters,
    normalize: bool = True,
    approx: str = "Rytov",
    load_fft: bool = False,
) -> tuple[NDArray, tuple]:
    """Reconstruct Electric field based on the given approximation

    Args:
        array (NDArray): acquired hologram
        ref_array (NDArray): acquired reference hologram
        params (ODTParameters): parameters for the ODT system
        normalize (bool, optional): noramalize the amplitude of the field. Defaults to True.
        approx (str, optional): Approximation for scattering. Defaults to "Rytov". "Rytov" and "Born" are available.

    Raises:
        ValueError: approx must be 'Born' or 'Rytov'

    Returns:
        NDArray: reconstructed electric field
        tuple: center of the reconstructed field
    """
    assert approx in ["Rytov", "Born"]
    global EDGE_SIZE

    E_initial = xp.ones(
        (2 * params.aperturesize + 1, 2 * params.aperturesize + 1), dtype=xp.complex128
    )

    # get off-axis interference term
    if load_fft:
        array_fft = array
    else:
        # array_fft = (params.pixelsize) ** 2 * xp.fft.fftshift(xp.fft.fft2(array))
        norm_array = array * params.pixelsize
        norm_array_fft = xp.fft.fftshift(xp.fft.fft2(norm_array, norm="ortho"))
        array_fft = norm_array_fft / params.k_unit
    disk = make_disk(params.offaxis_center, params.aperturesize // 2, array_fft.shape)
    array_fft = array_fft * disk
    max_x, max_y, _ = find_max_args(np.abs(array_fft))
    oblique_center = (
        max_x - params.offaxis_center[0],
        max_y - params.offaxis_center[1],
    )

    left_index = max_x - params.aperturesize
    right_index = max_x + params.aperturesize + 1
    top_index = max_y - params.aperturesize
    bottom_index = max_y + params.aperturesize + 1

    array_fft_pad = xp.pad(
        array_fft,
        (
            (params.aperturesize, params.aperturesize),
            (params.aperturesize, params.aperturesize),
        ),
        mode="constant",
        constant_values=0,
    )

    array_fft = array_fft_pad[
        left_index + params.aperturesize : right_index + params.aperturesize,
        top_index + params.aperturesize : bottom_index + params.aperturesize,
    ]

    # array_cropped = (
    #     params.freq_per_pixel**2
    #     * xp.fft.ifft2(xp.fft.ifftshift(array_fft))[EDGE_SIZE:, EDGE_SIZE:]
    # )
    norm_array_fft = array_fft * params.k_unit
    norm_array_cropped = xp.fft.ifft2(xp.fft.ifftshift(norm_array_fft), norm="ortho")[
        EDGE_SIZE:, EDGE_SIZE:
    ]
    array_cropped = norm_array_cropped / params.imgpx_unit

    array_cropped[0:2, :] = 1e-6
    array_cropped[:, 0:2] = 1e-6

    if load_fft:
        ref_array_fft = ref_array
    else:
        # ref_array_fft = (params.pixelsize**2) * xp.fft.fftshift(
        #     xp.fft.fft2(ref_array)
        # )
        norm_ref_array = ref_array * params.pixelsize
        norm_ref_array_fft = xp.fft.fftshift(xp.fft.fft2(norm_ref_array, norm="ortho"))
        ref_array_fft = norm_ref_array_fft / params.k_unit
    ref_array_fft = ref_array_fft * disk
    ref_array_fft_pad = xp.pad(
        ref_array_fft,
        (
            (params.aperturesize, params.aperturesize),
            (params.aperturesize, params.aperturesize),
        ),
        mode="constant",
        constant_values=0,
    )
    ref_array_fft = ref_array_fft_pad[
        left_index + params.aperturesize : right_index + params.aperturesize,
        top_index + params.aperturesize : bottom_index + params.aperturesize,
    ]
    # ref_array_cropped = (params.freq_per_pixel) ** 2 * xp.fft.ifft2(
    # xp.fft.ifftshift(ref_array_fft)
    # )[EDGE_SIZE:, EDGE_SIZE:]
    norm_ref_array_fft = ref_array_fft * params.k_unit
    norm_ref_array_cropped = xp.fft.ifft2(
        xp.fft.ifftshift(norm_ref_array_fft), norm="ortho"
    )[EDGE_SIZE:, EDGE_SIZE:]
    ref_array_cropped = norm_ref_array_cropped / params.imgpx_unit

    ref_array_cropped[0:2, :] = 1e-6
    ref_array_cropped[:, 0:2] = 1e-6
    # print(f"nan(array), {xp.count_nonzero(xp.isnan(array_cropped))}")
    # print(f"nan(ref), {xp.count_nonzero(xp.isnan(ref_array_cropped))}")
    # print(f"before approx, {xp.count_nonzero(xp.isnan(array_cropped))}")
    if approx == "Born":
        E_array = (array_cropped - ref_array_cropped) / ref_array_cropped
    elif approx == "Rytov":
        # div = array_cropped / ref_array_cropped
        # print(div.dtype)
        # print(f"div, {xp.count_nonzero(div==0)}")
        # print(f"div isinfinite, {xp.count_nonzero(~xp.isfinite(div))}")
        # div[div == 0] = 0.0001
        # E_array = ref_array_cropped * xp.log(div)
        # print(f"zero arr, {xp.count_nonzero(array_cropped==0)}")
        # print(f"zero ref, {xp.count_nonzero(ref_array_cropped==0)}")
        log_array = xp.log(array_cropped)
        # print(f"logarr isnan, {xp.count_nonzero(xp.isnan(log_array))}")
        # print(f"logarr isinfinite, {xp.count_nonzero(~xp.isfinite(log_array))}")
        log_ref_array = xp.log(ref_array_cropped)
        # print(f"logrefarr isnan, {xp.count_nonzero(xp.isnan(log_ref_array))}")
        # print(f"logrefarr isinfinite, {xp.count_nonzero(~xp.isfinite(log_ref_array))}")
        logdiv = log_array - log_ref_array
        # print(f"logdiv isnan, {xp.count_nonzero(xp.isnan(logdiv))}")
        # print(f"logdiv isinfinite, {xp.count_nonzero(~xp.isfinite(logdiv))}")
        E_array = logdiv
    else:
        raise ValueError("approx must be 'Born' or 'Rytov'")
    # print(f"after approx, {xp.count_nonzero(xp.isnan(E_array))}")

    # if normalize:
    #     array_cropped = array_cropped / ref_array_cropped

    return E_array, oblique_center


class ODTSynthesizer(Synthesizer):
    def ODT_synthesize(
        self, approx: str, hermite=False, load_fft=False
    ) -> tuple[np.ndarray, np.ndarray]:
        assert approx in ["Rytov", "Born"]
        assert self.reference_data is not None
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
            ),
            dtype=xp.complex128,
        )
        synthesized_center = (
            self.params.aperturesize - EDGE_SIZE // 2,
            self.params.aperturesize - EDGE_SIZE // 2,
            self.params.aperturesize - EDGE_SIZE // 2,
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        print("ODT Synthesizing...")
        for i in tqdm(range(len(self.target_data))):
            array = xp.load(self.target_data[i])
            ref_array = xp.load(self.reference_data[i])
            E_approx, oblique_center = reconstruct_E(
                array,
                ref_array,
                params=self.params,
                normalize=True,
                approx=approx,
                load_fft=load_fft,
            )

            # e_fft = (self.params.pixelsize**2) * xp.fft.fftshift(
            #     xp.fft.fft2(E_approx)
            # )
            norm_E_approx = E_approx * self.params.imgpx_unit
            norm_e_fft = xp.fft.fftshift(xp.fft.fft2(norm_E_approx, norm="ortho"))
            e_fft = norm_e_fft / self.params.k_unit
            disk_synthesized = make_disk(
                (
                    synthesized_center[0] - oblique_center[0],
                    synthesized_center[1] - oblique_center[1],
                ),
                self.params.aperturesize // 2,
                E_approx.shape,
            )

            e_fft_cropped = e_fft * disk_synthesized

            # make kz disk for scattering potential
            kz_i = np.sqrt(
                self.params.ki_mag**2
                - oblique_center[0] ** 2
                - oblique_center[1] ** 2
            )

            xx, yy = xp.meshgrid(
                xp.arange(2 * self.params.aperturesize + 1 - EDGE_SIZE),
                xp.arange(2 * self.params.aperturesize + 1 - EDGE_SIZE),
                indexing="ij",
            )
            disk = (xx - synthesized_center[0] - oblique_center[0]) ** 2 + (
                yy - synthesized_center[1] - oblique_center[1]
            ) ** 2
            disk[disk > (self.params.aperturesize // 2) ** 2] = 0
            kz_disk = xp.sqrt((self.params.aperturesize // 2) ** 2 - disk) + kz_i
            kz_disk[disk > (self.params.aperturesize // 2) ** 2] = 0
            kz_disk = kz_disk * self.params.k_unit

            # scatter_potential_fft = 2j * xp.pi * kz_disk * e_fft_cropped
            scatter_potential_fft = 2j * kz_disk * e_fft_cropped

            scatter_potential_fft3d = map_aperture_to_3Dkspace(
                scatter_potential_fft,
                synthesized_fft.shape,
                oblique_center,
                self.params.aperturesize,
                self.params,
            )

            synthesized_fft = synthesized_fft + scatter_potential_fft3d
            synthesized_weight += scatter_potential_fft3d != 0

            # if hermite:
            #     conjugate_scatter_potential_fft = xp.flipud(
            #         xp.fliplr(scatter_potential_fft.conjugate())
            #     )
            #     conjugate_sphere_center = (
            #         synthesized_center[0] + oblique_center[0],
            #         synthesized_center[1] + oblique_center[1],
            #         synthesized_center[2] + kz_i,
            #     )
            #     conjugate_sphere_mask = make_semisphere_surface(
            #         conjugate_sphere_center,
            #         self.params.ki_mag,
            #         synthesized_fft.shape,
            #         upper=False,
            #     )
            #     conjugate_scatter_potential_fft_tiled = xp.stack(
            #         [conjugate_scatter_potential_fft] * synthesized_fft.shape[2],
            #         axis=-1,
            #     )
            #     conjugate_scatter_potential_fft_tiled = (
            #         conjugate_scatter_potential_fft_tiled * conjugate_sphere_mask
            #     )
            #     synthesized_fft = (
            #         synthesized_fft + conjugate_scatter_potential_fft_tiled
            #     )
            #     synthesized_weight += conjugate_scatter_potential_fft_tiled != 0

        synthesized_weight -= synthesized_weight != 1
        synthesized_fft /= synthesized_weight
        # print(f"nan num(fft); {xp.count_nonzero(xp.isnan(synthesized_fft))}")

        # synthesized_array = self.params.pixelsize**3 * xp.fft.ifftn(
        #     xp.fft.ifftshift(synthesized_fft)
        # )
        norm_synthesized_fft = synthesized_fft * self.params.k_unit ** (3 / 2)
        norm_synthesized_array = xp.fft.ifftn(
            xp.fft.ifftshift(norm_synthesized_fft), norm="ortho"
        )
        synthesized_array = norm_synthesized_array / self.params.imgpx_unit ** (3 / 2)
        # print(f"nan num(space); {xp.count_nonzero(xp.isnan(synthesized_array))}")

        synthesized_array = xp.fft.fftshift(synthesized_array, axes=(2))

        return synthesized_array, synthesized_fft

    def iterative_ODT(
        self, approx, epsilon=1e-6, max_N=100, hermite=False, load_fft=False
    ):
        # principle: Fr < 0
        array3d, array3d_fft = self.ODT_synthesize(approx, hermite, load_fft)
        array3d = xp.fft.ifftshift(array3d)
        current_array = array3d.copy()
        former_array = current_array.copy()
        delta = xp.inf
        iteration = 0
        while (delta > epsilon) and (iteration < max_N):
            current_array[current_array > 0] = 0
            current_fft = xp.fft.fftshift(xp.fft.fftn(current_array))
            current_fft[array3d_fft != 0] = array3d_fft[array3d_fft != 0]
            current_array = xp.real(xp.fft.ifftn(xp.fft.ifftshift(current_fft)))

            delta = xp.sum(xp.abs(current_array - former_array))
            iteration += 1
            former_array = current_array.copy()

            print(f"delta: {delta}, iteration: {iteration}")

        current_array = xp.fft.fftshift(current_array, axes=(2))

        return current_array, current_fft


def map_aperture_to_3Dkspace(
    array: NDArray,
    shape: tuple[int, int, int],
    oblique_center: tuple[int, int],
    aperturesize: int,
    params: ODTParameters,
):
    """map 2d array to 3d array(ODT)

    Args:
        array (NDArray): 2D array to be projected. array size should be 2*aperturesize+1 square.
        shape (NDArray): shape of the output 3d array. xy shape must be consistent with the input array
        oblique_center (tuple): center position of the input circle
        aperturesize (int): aperture size of the input circle
        km (float): magnitude of the wave vector
    """
    # make kz index 2darray
    xx, yy = xp.meshgrid(
        xp.arange(2 * aperturesize + 1),
        xp.arange(2 * aperturesize + 1),
        indexing="ij",
    )

    kz_i = xp.sqrt(params.ki_mag**2 - oblique_center[0] ** 2 - oblique_center[1] ** 2)

    # inside the aperture
    mask = (
        (xx - params.aperturesize + oblique_center[0]) ** 2
        + (yy - params.aperturesize + oblique_center[1]) ** 2
    ) < (aperturesize // 2) ** 2

    kz_index_square = (
        params.ki_mag**2
        - (xx - params.aperturesize + oblique_center[0]) ** 2
        - (yy - params.aperturesize + oblique_center[1]) ** 2
    ) * mask

    kz_index_array = xp.sqrt(kz_index_square)

    KZ_index_array = kz_index_array - kz_i
    KZ_index_array = KZ_index_array * mask
    KZ_index_array = KZ_index_array.astype(int)

    # TODO: speed up later
    index_array = xp.zeros(shape, dtype=xp.complex128)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # index_array[i, j, KZ_index_array[i, j]] = KZ_value_array[i, j]
            index_array[i, j, KZ_index_array[i, j] + params.aperturesize] = 1

    array_tiled = xp.stack([array] * shape[2], axis=-1)
    array_projected = array_tiled * index_array
    return array_projected


def calc_refractive_index_square(array3d, params):
    r_3d_square = params.n_sol**2 * (
        xp.ones(array3d.shape, dtype=xp.complex128)
        - array3d / (params.ki_mag * params.k_unit) ** 2
    )
    # r_3d_square = params.n_sol**2 * (
    #     xp.ones(array3d.shape, dtype=xp.complex128) - array3d / (params.ki_mag) ** 2
    # )
    # print(f"nan num(refractive index); {xp.count_nonzero(xp.isnan(r_3d_square))}")
    return r_3d_square
