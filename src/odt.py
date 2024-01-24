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

        if self.NA_illumi is not None:
            self.ki_lateral_mag = self.ki_mag * self.NA_illumi / self.NA

    def print_all_parameters(self):
        super().print_all_parameters()
        print(f"{self.n_sol=}")
        print(f"{self.ki_mag=}")
        print(f"{self.ki_lateral_mag=}")


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
        array_fft = xp.fft.fftshift(xp.fft.fft2(array))
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

    array_cropped = xp.fft.ifft2(xp.fft.ifftshift(array_fft))[EDGE_SIZE:, EDGE_SIZE:]

    if load_fft:
        ref_array_fft = ref_array
    else:
        ref_array_fft = xp.fft.fftshift(xp.fft.fft2(ref_array))
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
    ref_array_cropped = xp.fft.ifft2(xp.fft.ifftshift(ref_array_fft))[
        EDGE_SIZE:, EDGE_SIZE:
    ]

    if approx == "Born":
        E_array = array_cropped - ref_array_cropped
    elif approx == "Rytov":
        E_array = ref_array_cropped * xp.log(array_cropped / ref_array_cropped)
    else:
        raise ValueError("approx must be 'Born' or 'Rytov'")

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

            e_fft = xp.fft.fftshift(xp.fft.fft2(E_approx))

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
        synthesized_fft /= synthesized_weight  # TODO
        synthesized_array = xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft))

        return synthesized_array, synthesized_fft

    def iterative_ODT(
        self, approx, epsilon=1e-6, max_N=100, hermite=False, load_fft=False
    ):
        # principle: Fr < 0
        array3d, array3d_fft = self.ODT_synthesize(approx, hermite, load_fft)
        # odt_array = xp.abs(calc_refractive_index_square(array3d, self.params)) ** 0.5
        current_array = array3d.copy()
        former_array = current_array.copy()
        delta = xp.inf
        iteration = 0
        while (delta > epsilon) and (iteration < max_N):
            current_array[current_array > 0] = 0
            current_fft = xp.fft.fftshift(xp.fft.fftn(current_array))
            current_fft[array3d_fft != 0] = array3d_fft[array3d_fft != 0]
            current_array = xp.real(xp.fft.ifftn(xp.fft.ifftshift(current_fft)))

            # calc_phase_diff = xp.angle(current_array / former_array)
            # mean_phase_diff = xp.mean(calc_phase_diff)
            # current_array = current_array * xp.exp(-1j * mean_phase_diff)

            delta = xp.sum(xp.abs(current_array - former_array))
            iteration += 1
            former_array = current_array.copy()

            print(f"delta: {delta}, iteration: {iteration}")

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
    # print(f"{km=}")
    # print(f"{oblique_center=}")

    kz_i = xp.sqrt(params.ki_mag**2 - oblique_center[0] ** 2 - oblique_center[1] ** 2)
    # print(f"{kz_i=}")
    # print(oblique_center)

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
    # KZ_value_array = kz_index_array - kz_i

    KZ_index_array = kz_index_array - kz_i
    KZ_index_array = KZ_index_array * mask
    KZ_index_array = KZ_index_array.astype(int)
    # print(f"{kz_i=}")
    # print(xp.count_nonzero(KZ_index_array > 0))
    # print(
    #     KZ_index_array[
    #         params.aperturesize
    #         + oblique_center[0]
    #         - 10 : params.aperturesize
    #         + oblique_center[0]
    #         + 10,
    #         params.aperturesize
    #         + oblique_center[1]
    #         - 10 : params.aperturesize
    #         + oblique_center[1]
    #         + 10,
    #     ]
    # )
    # print(KZ_index_array[params.aperturesize, params.aperturesize])

    # TODO: speed up later
    index_array = xp.zeros(shape, dtype=xp.complex128)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # index_array[i, j, KZ_index_array[i, j]] = KZ_value_array[i, j]
            index_array[i, j, KZ_index_array[i, j] + params.aperturesize] = 1

    array_tiled = xp.stack([array] * shape[2], axis=-1)
    array_projected = array_tiled * index_array
    return array_projected


def make_semisphere_surface(center, radius, array_shape, upper=True):
    """Returns sphere surface filled with 1.

    Args:
        center (tuple): center of the sphere surface
        radius (int): radius of the sphere
        array_shape (tuple): shape of the output 3d array

    Returns:
        xp.array: array whose sphere surface is filled with 1, otherwise 0.
    """

    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape, array_shape)
    xx, yy, zz = xp.meshgrid(
        xp.arange(array_shape[0]),
        xp.arange(array_shape[1]),
        xp.arange(array_shape[2]),
        indexing="ij",
    )
    sphere = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    if upper:
        sphere = (xp.abs(sphere - radius**2) < 6) & (
            zz > center[2]
        )  # TODO: 6 is a magic number
    else:
        sphere = (xp.abs(sphere - radius**2) < 6) & (zz < center[2])
    return sphere


def calc_refractive_index_square(array3d, params):
    r_3d_square = params.n_sol**2 * (
        xp.ones(array3d.shape, dtype=xp.complex128) - array3d / params.ki_mag**2
    )
    return r_3d_square
