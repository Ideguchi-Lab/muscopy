import os

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import numpy as np
from PIL import Image
from scipy import optimize
from scipy.interpolate import interp2d
from skimage.restoration import unwrap_phase
from tqdm import tqdm

from src.qpi import QPIParameters, make_disk


class Synthesizer:
    def __init__(self):
        pass

    def set_parameters(self, wavelength, NA, img_shape, img_center, pixelsize, offaxis_center):
        self.params = QPIParameters(wavelength, NA, img_shape, img_center, pixelsize, offaxis_center)
        self.params.calc_params()

    def set_offaxis_center(self, offaxis_center):
        self.params.offaxis_center = offaxis_center

    def set_path(self, target, reference):
        self.target_path = target
        self.reference_path = reference

    def load_oblique_centers(self, centers_path):
        self.oblique_centers = []
        with open(centers_path, "r") as f:
            for line in f:
                self.oblique_centers.append(tuple(map(int, line.split(","))))

    def search_centers(self, output_path=None):
        path_list = os.listdir(self.target_path)
        path_list.sort()
        pic_path_list = []
        for i in range(len(path_list)):
            if path_list[i].endswith(".png"):
                pic_path_list.append(path_list[i])

        oblique_centers_subpixel = []
        print("searching centers...")
        for i in tqdm(range(len(pic_path_list))):
            pic_path = pic_path_list[i]
            image = Image.open(self.target_path + pic_path)
            array = xp.array(image.getdata()).reshape(self.params.img_shape)
            array_fft = xp.fft.fftshift(xp.fft.fft2(array))

            disk = make_disk(self.params.offaxis_center, self.params.aperturesize / 2, array_fft.shape)
            array_fft = array_fft * disk
            array_fft = array_fft[
                self.params.offaxis_center[1]
                - self.params.aperturesize // 2 : self.params.offaxis_center[1]
                + self.params.aperturesize // 2
                + 1,
                self.params.offaxis_center[0]
                - self.params.aperturesize // 2 : self.params.offaxis_center[0]
                + self.params.aperturesize // 2
                + 1,
            ]

            max_x, max_y, max_value = find_max_args_subpixel(xp.asnumpy(xp.abs(array_fft)))
            max_x -= self.params.aperturesize // 2
            max_y -= self.params.aperturesize // 2
            oblique_centers_subpixel.append((max_x, max_y))

        fit_center, R, residu = circle_fitting(oblique_centers_subpixel)
        true_center = (self.params.offaxis_center[0] + fit_center[1], self.params.offaxis_center[1] + fit_center[0])

        if output_path is not None:
            with open(output_path + "centers.txt", "w") as f:
                for center in oblique_centers_subpixel:
                    f.write("{},{}\n".format(round(center[0]), round(center[1])))
        else:
            with open(self.target_path + "centers.txt", "w") as f:
                for center in oblique_centers_subpixel:
                    f.write("{},{}\n".format(round(center[0]), round(center[1])))

        return true_center, R, residu

    def synthesize(self, background_region=[(0, 50), (0, 50)]):
        synthesized_fft = xp.zeros(
            (2 * (self.params.aperturesize) + 1, 2 * (self.params.aperturesize) + 1), dtype=xp.complex128
        )
        synthesized_center = (self.params.aperturesize, self.params.aperturesize)
        synthesized_weight = xp.ones(synthesized_fft.shape)
        aperture_center = (self.params.aperturesize // 2, self.params.aperturesize // 2)

        path_list = os.listdir(self.target_path)
        pic_path_list = []
        ref_path_list = os.listdir(self.reference_path)
        ref_pic_path_list = []

        for i in range(len(path_list)):
            filename = path_list[i]
            if filename.endswith(".png"):
                pic_path_list.append(filename)
        for i in range(len(ref_path_list)):
            filename = ref_path_list[i]
            if filename.endswith(".png"):
                ref_pic_path_list.append(filename)

        pic_path_list.sort()
        ref_pic_path_list.sort()

        print("synthesizing...")
        for i in tqdm(range(len(pic_path_list))):
            filename = pic_path_list[i]
            ref_filename = ref_pic_path_list[i]
            image = Image.open(self.target_path + filename)
            ref_image = Image.open(self.reference_path + ref_filename)

            array = xp.array(image).reshape(self.params.img_shape)
            ref_array = xp.array(ref_image).reshape(self.params.img_shape)

            fft = xp.fft.fftshift(xp.fft.fft2(array))
            ref_fft = xp.fft.fftshift(xp.fft.fft2(ref_array))

            oblique_center = self.oblique_centers[i]
            disk = make_disk(self.params.offaxis_center, self.params.aperturesize / 2, fft.shape)

            fft = fft * disk
            ref_fft = ref_fft * disk

            left_index = self.params.offaxis_center[1] + oblique_center[0] - self.params.aperturesize
            right_index = self.params.offaxis_center[1] + oblique_center[0] + self.params.aperturesize + 1
            top_index = self.params.offaxis_center[0] + oblique_center[1] - self.params.aperturesize
            bottom_index = self.params.offaxis_center[0] + oblique_center[1] + self.params.aperturesize + 1
            if left_index < 0:
                left_index = 0
            if right_index > self.params.img_shape[0]:
                right_index = self.params.img_shape[0]
            if top_index < 0:
                top_index = 0
            if bottom_index > self.params.img_shape[1]:
                bottom_index = self.params.img_shape[1]

            fft = fft[
                left_index:right_index,
                top_index:bottom_index,
            ]
            ref_fft = ref_fft[
                left_index:right_index,
                top_index:bottom_index,
            ]

            if self.params.offaxis_center[1] + oblique_center[0] - self.params.aperturesize < 0:
                fft = xp.pad(
                    fft,
                    ((-self.params.offaxis_center[1] - oblique_center[0] + self.params.aperturesize, 0), (0, 0)),
                    "constant",
                )
                ref_fft = xp.pad(
                    ref_fft,
                    ((-self.params.offaxis_center[1] - oblique_center[0] + self.params.aperturesize, 0), (0, 0)),
                    "constant",
                )
            if (
                self.params.offaxis_center[1] + oblique_center[0] + self.params.aperturesize + 1
                > self.params.img_shape[0]
            ):
                fft = xp.pad(
                    fft,
                    (
                        (
                            0,
                            self.params.offaxis_center[1]
                            + oblique_center[0]
                            + self.params.aperturesize
                            + 1
                            - self.params.img_shape[0],
                        ),
                        (0, 0),
                    ),
                    "constant",
                )
                ref_fft = xp.pad(
                    ref_fft,
                    (
                        (
                            0,
                            self.params.offaxis_center[1]
                            + oblique_center[0]
                            + self.params.aperturesize
                            + 1
                            - self.params.img_shape[0],
                        ),
                        (0, 0),
                    ),
                    "constant",
                )
            if self.params.offaxis_center[0] + oblique_center[1] - self.params.aperturesize < 0:
                fft = xp.pad(
                    fft,
                    ((0, 0), (-self.params.offaxis_center[0] - oblique_center[1] + self.params.aperturesize, 0)),
                    "constant",
                )
                ref_fft = xp.pad(
                    ref_fft,
                    ((0, 0), (-self.params.offaxis_center[0] - oblique_center[1] + self.params.aperturesize, 0)),
                    "constant",
                )
            if (
                self.params.offaxis_center[0] + oblique_center[1] + self.params.aperturesize + 1
                > self.params.img_shape[1]
            ):
                fft = xp.pad(
                    fft,
                    (
                        (0, 0),
                        (
                            0,
                            self.params.offaxis_center[0]
                            + oblique_center[1]
                            + self.params.aperturesize
                            + 1
                            - self.params.img_shape[1],
                        ),
                    ),
                    "constant",
                )
                ref_fft = xp.pad(
                    ref_fft,
                    (
                        (0, 0),
                        (
                            0,
                            self.params.offaxis_center[0]
                            + oblique_center[1]
                            + self.params.aperturesize
                            + 1
                            - self.params.img_shape[1],
                        ),
                    ),
                    "constant",
                )

            qpi_array = xp.fft.ifft2(xp.fft.ifftshift(fft))
            ref_qpi_array = xp.fft.ifft2(xp.fft.ifftshift(ref_fft))
            qpi_divided = qpi_array / ref_qpi_array

            # normalize
            phase_backgound = xp.mean(
                xp.angle(
                    qpi_divided[
                        background_region[0][0] : background_region[0][1],
                        background_region[1][0] : background_region[1][1],
                    ]
                )
            )
            qpi_divided = qpi_divided * xp.exp(-1j * phase_backgound)

            amplitude_background = xp.mean(
                xp.abs(
                    qpi_divided[
                        background_region[0][0] : background_region[0][1],
                        background_region[1][0] : background_region[1][1],
                    ]
                )
            )
            qpi_divided = qpi_divided / amplitude_background

            fft_divided = xp.fft.fftshift(xp.fft.fft2(qpi_divided))

            disk_synthesized = make_disk(
                (synthesized_center[0] - oblique_center[1], synthesized_center[1] - oblique_center[0]),
                self.params.aperturesize // 2,
                synthesized_fft.shape,
            )

            fft_divided = fft_divided * disk_synthesized

            if i > 0:
                phase_offset = calc_phase_offset(fft_divided, synthesized_fft, synthesized_weight, synthesized_center)
                fft_divided = fft_divided * xp.exp(1j * phase_offset)

            synthesized_fft += fft_divided
            synthesized_weight += disk_synthesized != 0

        synthesized_fft /= synthesized_weight
        synthesized_qpi = xp.angle(xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft)))

        if _cp:
            synthesized_fft = xp.asnumpy(synthesized_fft)
            synthesized_qpi = xp.asnumpy(synthesized_qpi)

        synthesized_qpi = unwrap_phase(synthesized_qpi)

        return synthesized_qpi, synthesized_fft


def calc_phase_offset(array_fft, synthesized_fft, synthesized_weight, synthesized_center):
    # map array into expanded space
    # synthesized = xp.array(synthesized_fft)
    synthesized = xp.copy(synthesized_fft) / synthesized_weight
    overlap_region = (array_fft != 0) * (synthesized != 0)
    phase_offset = xp.mean(xp.angle(synthesized[overlap_region] / array_fft[overlap_region]))
    if _cp:
        phase_offset = xp.asnumpy(phase_offset)
    return phase_offset


def find_max_args_subpixel(array):
    # Create an interpolation function using interp2d
    x_data = np.arange(array.shape[1])
    y_data = np.arange(array.shape[0])
    z_data = array
    interp_func = interp2d(x_data, y_data, z_data, kind="cubic")

    # Define a grid for finer sampling
    x_fine = np.linspace(min(x_data), max(x_data), 1000)
    y_fine = np.linspace(min(y_data), max(y_data), 1000)

    # Calculate interpolated values on the fine grid
    z_interp = interp_func(x_fine, y_fine)

    # Find the maximum value and its position
    max_value = np.max(z_interp)
    max_idx = np.unravel_index(np.argmax(z_interp), z_interp.shape)
    max_x = x_fine[max_idx[0]]
    max_y = y_fine[max_idx[1]]

    return max_x, max_y, max_value


def circle_fitting(center_list: list) -> list:
    """
    center_list: [(x1, y1), (x2, y2), ...]
    """
    x = np.array([center[0] for center in center_list])
    y = np.array([center[1] for center in center_list])

    x_m = np.mean(x)
    y_m = np.mean(y)

    def calc_R(xc, yc):
        """calculate the distance of each 2D points from the center (xc, yc)"""
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    def f_2(c):
        """calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)"""
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(f_2, center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_R(*center_2)
    R_2 = Ri_2.mean()
    residu_2 = sum((Ri_2 - R_2) ** 2)

    return (xc_2, yc_2), R_2, residu_2
