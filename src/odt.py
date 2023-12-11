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


class ODTSynthesizer(Synthesizer):
    def ODT_synthesize(self) -> tuple[np.ndarray, np.ndarray]:
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
            if self.reference_data is not None:
                ref_array = xp.load(self.reference_data[i])
                array_cropped, oblique_center = preprocess_for_synthesis(
                    array, ref_array, params=self.params
                )

            else:
                array_cropped, oblique_center = preprocess_for_synthesis(
                    array, params=self.params
                )

            disk_synthesized = make_disk(
                (
                    synthesized_center[0] - oblique_center[1],
                    synthesized_center[1] - oblique_center[0],
                ),
                self.params.aperturesize // 2,
                array_cropped.shape,
            )

            fft_cropped = xp.fft.fftshift(xp.fft.fft2(array_cropped))

            fft_cropped = fft_cropped * disk_synthesized

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
            fft_cropped_tiled = xp.stack(
                [fft_cropped] * synthesized_fft.shape[2], axis=-1
            )
            fft_cropped_tiled = fft_cropped_tiled * sphere_mask
            synthesized_fft = synthesized_fft + fft_cropped_tiled
            synthesized_weight += fft_cropped_tiled != 0

        synthesized_fft /= synthesized_weight
        synthesized_array = xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft))

        # if _cp:
        #     synthesized_array = xp.asnumpy(synthesized_array)
        #     synthesized_fft = xp.asnumpy(synthesized_fft)

        return synthesized_array, synthesized_fft

    def iterative_ODT(self, epsilon=1e-6, max_N=100):
        ref_array, ref_fft = self.ODT_synthesize()
        current_array = ref_array.copy()
        current_odt = xp.angle(current_array)
        delta = xp.inf
        iteration = 0
        while (delta > epsilon) and (iteration < max_N):
            current_odt[current_odt < 0] = 0
            current_array = xp.abs(current_array) * xp.exp(1j * current_odt)
            current_fft = xp.fft.fftshift(xp.fft.fftn(current_array))
            current_fft[ref_fft != 0] = ref_fft[ref_fft != 0]
            current_array = xp.fft.ifftn(xp.fft.ifftshift(current_fft))

            # delta = xp.sum(xp.angle(current_array) - current_odt) #TODO: consider better delta
            iteration += 1
            current_odt = xp.angle(current_array)

            print(f"delta: {delta}, iteration: {iteration}")

        return current_array, current_fft


def map_to_3d(array, shape, oblique_center, synthesized_center, aperturesize, kz):
    array_3d = xp.tile(array, (1, 1, shape[2]))
    center_sphere = (
        synthesized_center[0] + oblique_center[0],
        synthesized_center[1] + oblique_center[1],
        kz,
    )
    radius = aperturesize // 2
    sphere_mask = make_sphere_surface(center_sphere, radius, shape)
    array_3d = array_3d * sphere_mask
    array_weight = array_3d != 0
    return array_3d, array_weight


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
    sphere = (xp.abs(sphere - radius**2) < 6) & (
        zz > center[2]
    )  # TODO: 6 is a magic number
    return sphere
