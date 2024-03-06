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

from src.aperture_synthesis import EDGE_SIZE, Synthesizer, preprocess_for_synthesis
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
        self.fi_mag = (
            self.n_sol / self.wav / self.freq_per_pixel
        )  # |k| in terms of pixel unit

        self.k_per_pixel = (
            2 * np.pi * self.freq_per_pixel
        )  # unit of k in terms of pixel unit
        self.imgpx_unit = (
            self.pixelsize * self.img_shape[0] / (2 * self.aperturesize + 1)
        )  # unit image pixel size on the cropped image plane

        if self.NA_illumi is not None:
            self.fi_lateral_mag = (
                self.fi_mag * self.NA_illumi / self.n_sol
            )  # |k_T| in terms of pixel unit

        self.fi_z = int(
            self.fi_mag * (1 - self.NA_illumi**2 / self.n_sol**2) ** 0.5
        )  # |k_z| in terms of pixel unit

        self.fz_extent = int(
            (self.fi_mag - self.fi_z)
        )  # extent of kz axis in terms of pixel unit
        self.imgpx_unit_z = self.imgpx_unit * (
            (2 * self.aperturesize + 1) / (2 * self.fz_extent + 1)
        )  # unit image pixel size along z axis on the cropped image plane

    def print_all_parameters(self):
        super().print_all_parameters()
        print(f"{self.n_sol=}")
        print(f"{self.fi_mag=}")
        print(f"{self.fi_lateral_mag=}")
        print(f"{self.k_per_pixel=}")
        print(f"{self.imgpx_unit=}")
        print(f"{self.imgpx_unit_z=}")
        print(f"{self.fz_extent=}")
        print(f"{self.fi_z=}")


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
        norm_array = array * params.pixelsize
        norm_array_fft = xp.fft.fftshift(xp.fft.fft2(norm_array, norm="backward"))
        array_fft = norm_array_fft / params.k_per_pixel
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

    norm_array_fft = array_fft * params.k_per_pixel
    norm_array_cropped = xp.fft.ifft2(xp.fft.ifftshift(norm_array_fft), norm="backward")
    array_cropped = norm_array_cropped / params.imgpx_unit

    # array_cropped[0:EDGE_SIZE, :] = 1e-6
    # array_cropped[:, 0:EDGE_SIZE] = 1e-6

    if load_fft:
        ref_array_fft = ref_array
    else:
        norm_ref_array = ref_array * params.pixelsize
        norm_ref_array_fft = xp.fft.fftshift(
            xp.fft.fft2(norm_ref_array, norm="backward")
        )
        ref_array_fft = norm_ref_array_fft / params.k_per_pixel
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
    norm_ref_array_fft = ref_array_fft * params.k_per_pixel
    norm_ref_array_cropped = xp.fft.ifft2(
        xp.fft.ifftshift(norm_ref_array_fft), norm="backward"
    )
    ref_array_cropped = norm_ref_array_cropped / params.imgpx_unit

    # ref_array_cropped[0:EDGE_SIZE, :] = 1e-6
    # ref_array_cropped[:, 0:EDGE_SIZE] = 1e-6
    if approx == "Born":
        E_array = (array_cropped - ref_array_cropped) / ref_array_cropped
    elif approx == "Rytov":
        log_array = xp.log(array_cropped)
        log_ref_array = xp.log(ref_array_cropped)
        logdiv = log_array - log_ref_array
        E_array = logdiv
    else:
        raise ValueError("approx must be 'Born' or 'Rytov'")

    # if normalize:
    #     array_cropped = array_cropped / ref_array_cropped

    return E_array, oblique_center


class ODTSynthesizer(Synthesizer):
    def ODT_synthesize(
        self, approx: str, hermite=False, load_fft=False, calc_ocupancy=False
    ) -> tuple[np.ndarray, np.ndarray]:
        assert approx in ["Rytov", "Born"]
        assert self.reference_data is not None
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1,
                2 * (self.params.aperturesize) + 1,
                2 * self.params.fz_extent + 1,
            ),
            dtype=xp.complex128,
        )
        synthesized_center = (
            self.params.aperturesize,
            self.params.aperturesize,
            self.params.fz_extent,
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

            norm_E_approx = E_approx * self.params.imgpx_unit
            norm_e_fft = xp.fft.fftshift(xp.fft.fft2(norm_E_approx, norm="backward"))
            e_fft = norm_e_fft / self.params.k_per_pixel
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
            fz_i = np.sqrt(
                self.params.fi_mag**2 - oblique_center[0] ** 2 - oblique_center[1] ** 2
            )

            xx, yy = xp.meshgrid(
                xp.arange(2 * self.params.aperturesize + 1),
                xp.arange(2 * self.params.aperturesize + 1),
                indexing="ij",
            )
            disk = (xx - synthesized_center[0] + oblique_center[0]) ** 2 + (
                yy - synthesized_center[1] + oblique_center[1]
            ) ** 2  # L2 distance from the center of illumination vector
            disk_mask = disk < (self.params.aperturesize // 2) ** 2
            # disk[disk > (self.params.aperturesize // 2) ** 2] = 0
            # fz_disk = xp.sqrt(self.params.fi_mag**2 - disk)
            fz_disk = self.params.fi_mag**2 - disk
            # fz_disk[disk > (self.params.aperturesize // 2) ** 2] = 0
            fz_disk[fz_disk < 0] = 0
            fz_disk = fz_disk**0.5
            kz_disk = fz_disk * self.params.k_per_pixel * disk_mask
            # print(kz_disk.shape)
            # print(
            #     "norm of oblique shift", oblique_center[0] ** 2 + oblique_center[1] ** 2
            # )
            # print("nonzero", xp.sum(kz_disk != 0))
            # print("average", xp.mean(kz_disk[kz_disk != 0]) / self.params.k_per_pixel)

            scatter_potential_fft = 2j * kz_disk * e_fft_cropped

            scatter_potential_fft3d = map_aperture_to_3Dkspace(
                scatter_potential_fft,
                synthesized_fft.shape,
                oblique_center,
                self.params,
            )

            synthesized_fft = synthesized_fft + scatter_potential_fft3d
            synthesized_weight += scatter_potential_fft3d != 0

            if hermite:

                conj_fft3d = xp.flip(scatter_potential_fft3d, axis=(0, 1, 2))
                conj_fft3d = xp.conjugate(conj_fft3d)

                synthesized_fft = synthesized_fft + conj_fft3d
                synthesized_weight += conj_fft3d != 0

        synthesized_weight -= synthesized_weight != 1
        synthesized_fft /= synthesized_weight

        # calculate the volume of filled pixels
        filled_volume = xp.sum(synthesized_weight > 1)
        full_volume = calc_full_volume(self.params)
        ocupancy = filled_volume / full_volume
        print(f"ocupancy: {ocupancy}")

        norm_synthesized_fft = synthesized_fft * self.params.k_per_pixel ** (3 / 2)
        norm_synthesized_array = xp.fft.ifftn(
            xp.fft.ifftshift(norm_synthesized_fft), norm="backward"
        )
        synthesized_array = norm_synthesized_array / (
            self.params.imgpx_unit * self.params.imgpx_unit_z**0.5
        )

        synthesized_array = xp.fft.fftshift(synthesized_array, axes=(2))

        if calc_ocupancy:
            return synthesized_array, synthesized_fft, ocupancy
        else:
            return synthesized_array, synthesized_fft

    def iterative_ODT(
        self, approx, epsilon=1e-6, max_N=100, hermite=False, load_fft=False
    ):
        # principle: Fr < 0
        array3d, array3d_fft = self.ODT_synthesize(approx, hermite, load_fft)
        array3d = xp.fft.ifftshift(array3d, axes=(2))
        current_array = array3d.copy()
        former_array = current_array.copy()
        delta = xp.inf
        iteration = 0
        # start iteration
        while (delta > epsilon) and (iteration < max_N):
            current_array[xp.real(current_array) > 0] = 0
            norm_current_array = current_array * (
                self.params.imgpx_unit * self.params.imgpx_unit_z**0.5
            )  # normalization for fft
            norm_current_fft = xp.fft.fftshift(
                xp.fft.fftn(norm_current_array, norm="backward")
            )
            current_fft = norm_current_fft / self.params.k_per_pixel ** (3 / 2)
            current_fft[array3d_fft != 0] = array3d_fft[
                array3d_fft != 0
            ]  # substitute the measured value
            norm_current_fft = current_fft * self.params.k_per_pixel ** (3 / 2)
            norm_current_array = xp.fft.ifftn(
                xp.fft.ifftshift(norm_current_fft), norm="backward"
            )
            # norm_current_array = xp.real(
            #     norm_current_array
            # )  # cut off the imaginary part
            current_array = norm_current_array / (
                self.params.imgpx_unit * self.params.imgpx_unit_z**0.5
            )

            delta = xp.sum(
                xp.abs(current_array - former_array)
            )  # calculate the difference
            iteration += 1
            former_array = current_array.copy()

            print(f"delta: {delta}, iteration: {iteration}")

        current_array = xp.fft.fftshift(current_array, axes=(2))

        return current_array, current_fft


def map_aperture_to_3Dkspace(
    array: NDArray,
    shape: tuple[int, int, int],
    oblique_center: tuple[int, int],
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
    oblique_shift = oblique_center
    xx, yy = xp.meshgrid(
        xp.arange(shape[0]),
        xp.arange(shape[1]),
        indexing="ij",
    )
    _, _, zz = xp.meshgrid(
        xp.arange(shape[0]),
        xp.arange(shape[1]),
        xp.arange(shape[2]),
        indexing="ij",
    )
    circle = (xx - params.aperturesize + oblique_shift[0]) ** 2 + (
        yy - params.aperturesize + oblique_shift[1]
    ) ** 2
    circle = circle < (params.aperturesize // 2) ** 2
    Fz_circle = xp.sqrt(
        params.fi_mag**2
        - (xx - params.aperturesize + oblique_shift[0]) ** 2
        - (yy - params.aperturesize + oblique_shift[1]) ** 2
    ) - xp.sqrt(params.fi_mag**2 - oblique_shift[0] ** 2 - oblique_shift[1] ** 2)

    Fz_value = (Fz_circle + shape[2] // 2) * circle
    Fz_tile = xp.tile(Fz_value, (shape[2], 1, 1))
    Fz_tile = Fz_tile.transpose(1, 2, 0)

    Fz_tile = Fz_tile.astype(xp.int64)

    Fz_tile -= Fz_tile == 0  # to avoid 0 index match with zz

    Fz_index = zz == Fz_tile

    array_tiled = xp.stack([array] * shape[2], axis=-1)
    array_projected = array_tiled * Fz_index
    return array_projected


def calc_refractive_index_square(array3d, params):
    r_3d_square = params.n_sol**2 * (
        xp.ones(array3d.shape, dtype=xp.complex128)
        - array3d / (params.fi_mag * params.k_per_pixel) ** 2
    )
    return r_3d_square


def discard_z(array, threshold):
    array = array[:, :, threshold : array.shape[2] - threshold]
    return array


def discard_higher_kz(array, threshold):
    norm_factor = array.shape[2]
    array_fft = xp.fft.fftshift(xp.fft.fftn(array, norm="backward"))
    discarded = discard_z(array_fft, threshold)
    new_array = (
        xp.fft.ifftn(xp.fft.ifftshift(discarded), norm="forward") / norm_factor**3
    )
    return new_array


def zeropad_higher_kz(array, extend):
    norm_factor = array.shape[0] * array.shape[1] * array.shape[2]
    array_fft = xp.fft.fftshift(xp.fft.fftn(array, norm="backward"))
    new_array_fft = xp.zeros(
        (
            array_fft.shape[0],
            array_fft.shape[1],
            array_fft.shape[2] + extend * 2,
        ),
        dtype=xp.complex128,
    )
    new_array_fft[
        :,
        :,
        extend : array_fft.shape[2] + extend,
    ] = array_fft
    new_array = (
        xp.fft.ifftn(xp.fft.ifftshift(new_array_fft), norm="forward") / norm_factor
    )
    return new_array


def calc_full_volume(params):
    """Calculate the volume which can be filled with FW light under the given parameters

    Args:
        params (ODTParameters): parameters for the ODT system

    Returns:
        xp.float : Volume
    """
    theta = xp.arcsin(params.aperturesize / (2 * params.fi_mag))
    S = 2 * theta * params.fi_mag**2 - params.aperturesize * params.fi_mag * xp.cos(
        theta
    )
    V = S * xp.pi * params.aperturesize
    return V


def calc_normalized_L2error(array, ref_array):
    error = xp.sum(xp.abs(array - ref_array)) / xp.sum(xp.abs(ref_array))
    return error
