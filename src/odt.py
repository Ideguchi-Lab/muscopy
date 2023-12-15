import os
import sys

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import numpy as np
from tqdm import tqdm

from src.aperture_synthesis import Synthesizer, preprocess_for_synthesis, EDGE_SIZE
from src.qpi import QPIParameters, make_disk, qpi

EDGE_SIZE = 2  # for avoiding edge artifact in ifft


def find_max_args(array):
    max_value = xp.max(array)
    max_idx = xp.unravel_index(np.argmax(array), array.shape)
    max_x = max_idx[0]
    max_y = max_idx[1]

    return max_x, max_y, max_value


class ODTParameters(QPIParameters):
    def __init__(
        self, wavelength, NA, img_shape, img_center, pixelsize, offaxis_center, n_sol
    ):
        super().__init__(
            wavelength, NA, img_shape, img_center, pixelsize, offaxis_center
        )
        self.n_sol = n_sol

    def calc_params(self):
        super().calc_params()
        self.ki_mag = self.n_sol / self.wav / self.freq_per_pixel

    def print_all_parameters(self):
        super().print_all_parameters()
        print(f"{self.n_sol=}")
        print(f"{self.ki_mag=}")


def reconstruct_E(
    array: xp.array,
    ref_array: xp.array,
    params: ODTParameters,
    normalize=True,
    approx="Rytov",
) -> [xp.array, tuple]:
    assert approx in ["Rytov", "Born"]
    global EDGE_SIZE

    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    disk = make_disk(params.offaxis_center, params.aperturesize // 2, array_fft.shape)
    array_fft = array_fft * disk
    max_x, max_y, _ = find_max_args(np.abs(array_fft))
    oblique_center = (
        max_x - params.offaxis_center[1],
        max_y - params.offaxis_center[0],
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
        array_cropped = array_cropped - ref_array_cropped
    elif approx == "Rytov":
        array_cropped = ref_array_cropped * xp.log(array_cropped / ref_array_cropped)
    else:
        raise ValueError("approx must be 'Born' or 'Rytov'")

    if normalize:
        array_cropped = array_cropped / ref_array_cropped

    return array_cropped, oblique_center


class ODTSynthesizer(Synthesizer):
    def ODT_synthesize(
        self, approx: str, hermite=False
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
                array, ref_array, params=self.params, normalize=True, approx=approx
            )

            e_fft = xp.fft.fftshift(xp.fft.fft2(E_approx))

            disk_synthesized = make_disk(
                (
                    synthesized_center[0] - oblique_center[1],
                    synthesized_center[1] - oblique_center[0],
                ),
                self.params.aperturesize // 2,
                E_approx.shape,
            )

            e_fft_cropped = e_fft * disk_synthesized

            scatter_potential_fft = 2j * xp.pi * e_fft_cropped

            kz = np.sqrt(
                self.params.ki_mag**2
                - oblique_center[0] ** 2
                - oblique_center[1] ** 2
            )
            sphere_center = (
                synthesized_center[0] - oblique_center[1],
                synthesized_center[1] - oblique_center[0],
                synthesized_center[2] - kz,
            )
            sphere_mask = make_semisphere_surface(
                sphere_center, self.params.ki_mag, synthesized_fft.shape
            )
            # fft_cropped_tiled = xp.tile(fft_cropped, (1, 1, synthesized_fft.shape[2]))
            scatter_potential_fft_tiled = xp.stack(
                [scatter_potential_fft] * synthesized_fft.shape[2], axis=-1
            )
            scatter_potential_fft_tiled = scatter_potential_fft_tiled * sphere_mask
            synthesized_fft = synthesized_fft + scatter_potential_fft_tiled
            synthesized_weight += scatter_potential_fft_tiled != 0

        synthesized_fft /= synthesized_weight
        synthesized_array = xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft))

        # if _cp:
        #     synthesized_array = xp.asnumpy(synthesized_array)
        #     synthesized_fft = xp.asnumpy(synthesized_fft)

        return synthesized_array, synthesized_fft

    def iterative_ODT(self, approx, epsilon=1e-6, max_N=100):
        array3d, array3d_fft = self.ODT_synthesize(approx)
        current_array = array3d.copy()
        former_array = current_array.copy()
        delta = xp.inf
        iteration = 0
        while (delta > epsilon) and (iteration < max_N):
            current_array[xp.real(current_array) < 0] = 0
            current_fft = xp.fft.fftshift(xp.fft.fftn(current_array))
            current_fft[array3d_fft != 0] = array3d_fft[array3d_fft != 0]
            current_array = xp.fft.ifftn(xp.fft.ifftshift(current_fft))

            delta = xp.sum(xp.abs(current_array - former_array))
            iteration += 1
            former_array = current_array.copy()

            print(f"delta: {delta}, iteration: {iteration}")

        return current_array, current_fft


def map_to_3d(array, shape, oblique_center, synthesized_center, aperturesize, kz):
    pass


def make_semisphere_surface(center, radius, array_shape):
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
        indexing="xy",
    )
    sphere = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2
    kz_value = (zz - array_shape[2] // 2) + radius
    sphere = (xp.abs(sphere - radius**2) < 6) & (
        zz > center[2]
    )  # TODO: 6 is a magic number
    sphere *= kz_value
    return sphere


def calc_refractive_index_square(array3d, params):
    r_3d_square = params.n_sol**2(
        xp.ones(array3d.shape) - array3d / params.ki_mag**2
    )
    return r_3d_square
