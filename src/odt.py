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
                synthesized_center[2] + kz,
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
        synthesized_odt = xp.angle(xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft)))

        if _cp:
            synthesized_fft = xp.asnumpy(synthesized_fft)
            synthesized_odt = xp.asnumpy(synthesized_odt)

        return synthesized_odt, synthesized_fft

    def iterative_ODT(self, epsilon=1e-6, man_N=1000):
        pass


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
    sphere = (sphere < radius**2) & (zz > center[2])
    return sphere


# class ODTSynthesizer(Synthesizer):
#     def ODT_synthesize(self):
#         # calculate Z dimension
#         z_plus = self.params.ki_mag - np.sqrt(
#             self.params.ki_mag**2 - (self.params.aperturesize // 2) ** 2
#         )
#         z_minus = -self.params.ki_mag - np.sqrt(
#             self.params.ki_mag**2 - (self.params.aperturesize // 2) ** 2
#         )
#         z_length = int(np.ceil(z_plus - z_minus))
#         z_center = round(-z_minus)

#         memory_per_picture = (
#             (2 * self.params.aperturesize + 1) ** 2 * z_length * 16 * 1e-9
#         )
#         print("memory per picture: {} GB".format(memory_per_picture))

#         synthesized_fft = xp.zeros(
#             (
#                 2 * self.params.aperturesize + 1,
#                 2 * self.params.aperturesize + 1,
#                 z_length,
#             ),
#             dtype=xp.complex128,
#         )
#         synthesized_center = (
#             self.params.aperturesize,
#             self.params.aperturesize,
#             z_center,
#         )
#         synthesized_weight = xp.ones(
#             (
#                 2 * self.params.aperturesize + 1,
#                 2 * self.params.aperturesize + 1,
#                 z_length,
#             ),
#             dtype=xp.int64,
#         )

#         path_list = os.listdir(self.target_path)
#         pic_path_list = []
#         ref_path_list = os.listdir(self.ref_path)
#         ref_pic_path_list = []

#         for i in range(len(path_list)):
#             filename = path_list[i]
#             if filename.endswith(".png"):
#                 pic_path_list.append(filename)
#         for i in range(len(ref_path_list)):
#             filename = ref_path_list[i]
#             if filename.endswith(".png"):
#                 ref_pic_path_list.append(filename)

#         pic_path_list.sort()
#         ref_pic_path_list.sort()

#         print("ODT Synthesizing...")
#         for i in tqdm(range(len(pic_path_list))):
#             filename = pic_path_list[i]
#             ref_filename = ref_pic_path_list[i]
#             array = xp.array(
#                 Image.open(os.path.join(self.target_path, filename))
#             ).reshape(self.params.img_shape)
#             ref_array = xp.array(
#                 Image.open(os.path.join(self.ref_path, ref_filename))
#             ).reshape(self.params.img_shape)

#             fft = xp.fft.fftshift(xp.fft.fftn(array))
#             ref_fft = xp.fft.fftshift(xp.fft.fftn(ref_array))

#             oblique_center = self.oblique_centers[i]
#             disk = make_disk(
#                 self.params.off_axis,
#                 self.params.aperturesize / 2,
#                 self.params.img_shape,
#             )

#             fft = fft * disk
#             ref_fft = ref_fft * disk

#             left_index = (
#                 self.params.offaxis_center[1]
#                 + oblique_center[0]
#                 - self.params.aperturesize
#             )
#             right_index = (
#                 self.params.offaxis_center[1]
#                 + oblique_center[0]
#                 + self.params.aperturesize
#                 + 1
#             )
#             top_index = (
#                 self.params.offaxis_center[0]
#                 + oblique_center[1]
#                 - self.params.aperturesize
#             )
#             bottom_index = (
#                 self.params.offaxis_center[0]
#                 + oblique_center[1]
#                 + self.params.aperturesize
#                 + 1
#             )
#             if left_index < 0:
#                 left_index = 0
#             if right_index > self.params.img_shape[0]:
#                 right_index = self.params.img_shape[0]
#             if top_index < 0:
#                 top_index = 0
#             if bottom_index > self.params.img_shape[1]:
#                 bottom_index = self.params.img_shape[1]

#             fft = fft[
#                 left_index:right_index,
#                 top_index:bottom_index,
#             ]
#             ref_fft = ref_fft[
#                 left_index:right_index,
#                 top_index:bottom_index,
#             ]

#             if (
#                 self.params.offaxis_center[1]
#                 + oblique_center[0]
#                 - self.params.aperturesize
#                 < 0
#             ):
#                 fft = xp.pad(
#                     fft,
#                     (
#                         (
#                             -self.params.offaxis_center[1]
#                             - oblique_center[0]
#                             + self.params.aperturesize,
#                             0,
#                         ),
#                         (0, 0),
#                     ),
#                     "constant",
#                 )
#                 ref_fft = xp.pad(
#                     ref_fft,
#                     (
#                         (
#                             -self.params.offaxis_center[1]
#                             - oblique_center[0]
#                             + self.params.aperturesize,
#                             0,
#                         ),
#                         (0, 0),
#                     ),
#                     "constant",
#                 )
#             if (
#                 self.params.offaxis_center[1]
#                 + oblique_center[0]
#                 + self.params.aperturesize
#                 + 1
#                 > self.params.img_shape[0]
#             ):
#                 fft = xp.pad(
#                     fft,
#                     (
#                         (
#                             0,
#                             self.params.offaxis_center[1]
#                             + oblique_center[0]
#                             + self.params.aperturesize
#                             + 1
#                             - self.params.img_shape[0],
#                         ),
#                         (0, 0),
#                     ),
#                     "constant",
#                 )
#                 ref_fft = xp.pad(
#                     ref_fft,
#                     (
#                         (
#                             0,
#                             self.params.offaxis_center[1]
#                             + oblique_center[0]
#                             + self.params.aperturesize
#                             + 1
#                             - self.params.img_shape[0],
#                         ),
#                         (0, 0),
#                     ),
#                     "constant",
#                 )
#             if (
#                 self.params.offaxis_center[0]
#                 + oblique_center[1]
#                 - self.params.aperturesize
#                 < 0
#             ):
#                 fft = xp.pad(
#                     fft,
#                     (
#                         (0, 0),
#                         (
#                             -self.params.offaxis_center[0]
#                             - oblique_center[1]
#                             + self.params.aperturesize,
#                             0,
#                         ),
#                     ),
#                     "constant",
#                 )
#                 ref_fft = xp.pad(
#                     ref_fft,
#                     (
#                         (0, 0),
#                         (
#                             -self.params.offaxis_center[0]
#                             - oblique_center[1]
#                             + self.params.aperturesize,
#                             0,
#                         ),
#                     ),
#                     "constant",
#                 )
#             if (
#                 self.params.offaxis_center[0]
#                 + oblique_center[1]
#                 + self.params.aperturesize
#                 + 1
#                 > self.params.img_shape[1]
#             ):
#                 fft = xp.pad(
#                     fft,
#                     (
#                         (0, 0),
#                         (
#                             0,
#                             self.params.offaxis_center[0]
#                             + oblique_center[1]
#                             + self.params.aperturesize
#                             + 1
#                             - self.params.img_shape[1],
#                         ),
#                     ),
#                     "constant",
#                 )
#                 ref_fft = xp.pad(
#                     ref_fft,
#                     (
#                         (0, 0),
#                         (
#                             0,
#                             self.params.offaxis_center[0]
#                             + oblique_center[1]
#                             + self.params.aperturesize
#                             + 1
#                             - self.params.img_shape[1],
#                         ),
#                     ),
#                     "constant",
#                 )

#             qpi_array = xp.fft.ifft2(xp.fft.ifftshift(fft))
#             ref_qpi_array = xp.fft.ifft2(xp.fft.ifftshift(ref_fft))
#             qpi_divided = qpi_array / ref_qpi_array

#             # normalize
#             phase_backgound = xp.mean(
#                 xp.angle(
#                     qpi_divided[
#                         background_region[0][0] : background_region[0][1],
#                         background_region[1][0] : background_region[1][1],
#                     ]
#                 )
#             )
#             qpi_divided = qpi_divided * xp.exp(-1j * phase_backgound)

#             amplitude_background = xp.mean(
#                 xp.abs(
#                     qpi_divided[
#                         background_region[0][0] : background_region[0][1],
#                         background_region[1][0] : background_region[1][1],
#                     ]
#                 )
#             )
#             qpi_divided = qpi_divided / amplitude_background

#             fft_divided = xp.fft.fftshift(xp.fft.fft2(qpi_divided))

#             disk_synthesized = make_disk(
#                 (
#                     synthesized_center[0] - oblique_center[1],
#                     synthesized_center[1] - oblique_center[0],
#                 ),
#                 self.params.aperturesize // 2,
#                 synthesized_fft.shape,
#             )

#             fft_divided = fft_divided * disk_synthesized

#             kz_i = xp.sqrt(
#                 self.params.ki_mag**2
#                 - oblique_center[0] ** 2
#                 - oblique_center[1] ** 2
#             )
#             fft_3d, array_weight = map_to_3d(
#                 fft_divided,
#                 synthesized_fft.shape,
#                 oblique_center,
#                 synthesized_center,
#                 self.params.aperturesize,
#                 kz_i,
#             )

#             synthesized_fft = synthesized_fft + fft_3d
#             synthesized_weight = synthesized_weight + array_weight

#         synthesized_weight -= synthesized_weight != 1
#         synthesized_fft = synthesized_fft / synthesized_weight
#         synthesized_array = xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft))

#         synthesized_odt = xp.angle(synthesized_array)

#         if _cp:
#             synthesized_odt = xp.asnumpy(synthesized_odt)
#             synthesized_fft = xp.asnumpy(synthesized_fft)

#         return synthesized_odt, synthesized_fft
