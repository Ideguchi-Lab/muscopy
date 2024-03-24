import os

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from muscopy.cfg import EDGE_SIZE
from muscopy.qpi import QPIParameters, correct_offset, make_disk


def find_max_args(array: xp.array) -> tuple[int, int, float]:
    """Find maximum value and its index in the array

    Args:
        array (xp.array): array to find maximum value

    Returns:
        tuple[int, int, float]: position of max value, and max value itself
    """
    max_value = xp.max(array)
    max_idx = xp.unravel_index(np.argmax(array), array.shape)
    max_x = max_idx[0]
    max_y = max_idx[1]

    return max_x, max_y, max_value


def crop_oblique_array(
    array: xp.array, oblique_center: tuple[int, int], aperturesize: int
) -> tuple[xp.array, tuple[int, int]]:
    """Crop the array with the oblique center and aperturesize

    Args:
        array (xp.array): input array
        oblique_center (tuple[int, int]): center of the oblique array
        aperturesize (int): size of the aperturesize

    Returns:
        xp.array: cropped array
        tuple[int, int]: oblique shift in the Fourier space
    """
    max_x, max_y, _ = find_max_args(xp.abs(array))
    oblique_shift = (max_x - oblique_center[0], max_y - oblique_center[1])

    left_index = max_x - aperturesize
    right_index = max_x + aperturesize + 1
    top_index = max_y - aperturesize
    bottom_index = max_y + aperturesize + 1

    array_pad = xp.pad(
        array,
        (
            (aperturesize, aperturesize),
            (aperturesize, aperturesize),
        ),
        mode="constant",
        constant_values=0,
    )

    array_cropped = array_pad[
        left_index + aperturesize : right_index + aperturesize,
        top_index + aperturesize : bottom_index + aperturesize,
    ]

    return array_cropped, oblique_shift


def get_oblique_field(array: xp.array, params: QPIParameters) -> tuple[xp.array, tuple[int, int]]:
    """internal method. get the electric field from the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class

    Returns:
        xp.array: field from the hologram array
        tuple[int, int]: oblique shift in the Fourier space
    """
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    mask = make_disk(params.offaxis_center, params.aperturesize / 2, params.img_shape)
    array_fft = array_fft * mask

    array_fft, oblique_shift = crop_oblique_array(array_fft, params.offaxis_center, params.aperturesize)
    # remove EDGE to avoid the edge effect
    array_field = xp.fft.ifft2(xp.fft.ifftshift(array_fft))[EDGE_SIZE:, EDGE_SIZE:]

    return array_field, oblique_shift


def preprocess_for_synthesis(
    array: xp.array,
    ref_array: xp.array,
    params: QPIParameters,
    offset_regs: list[tuple[tuple[int, int], tuple[int, int]]] | None = None,
    crop_center: bool = False,
    c_r: int = 5,
) -> tuple[xp.array, xp.array]:
    """internal method. preprocess the array for synthesis

    Args:
        array (xp.array): input array
        ref_array (xp.array): reference array
        params (QPIParameters): QPIParameters class
        offset_regs (list[tuple[tuple[int, int], tuple[int, int]]], optional): offset regions. Defaults to None.
        crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
        c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

    Returns:
        tuple[xp.array, xp.array]: preprocessed array and its Fourier transform
    """

    array_field, oblique_shift = get_oblique_field(array, params)
    ref_array_field, _ = get_oblique_field(ref_array, params)

    array_div = array_field / ref_array_field

    if offset_regs is not None:
        array_div = correct_offset(array_div, offset_regs)

    array_div_fft = xp.fft.fftshift(xp.fft.fft2(array_div))
    disk_for_synthesis = make_disk(
        (
            params.aperturesize - oblique_shift[0],
            params.aperturesize - oblique_shift[1],
        ),
        params.aperturesize // 2,
        array_field.shape,
    )

    array_div_fft = array_div_fft * disk_for_synthesis

    # for mipqpi
    if crop_center:
        mask_highpass = make_disk(
            (params.aperturesize - oblique_shift[0], params.aperturesize - oblique_shift[1]),
            c_r,
            array_div.shape,
            highpass=True,
        )
        array_div_fft = array_div_fft * mask_highpass

    return array_div, array_div_fft


class Synthesizer:
    def __init__(self, params: QPIParameters, target: list[xp.array], reference: list[xp.array]):
        """Synthesizer class for aperture synthesis

        Args:
            params (QPIParameters): QPIParameters class
            target (list[xp.array]): sample holograms
            reference (list[xp.array]): reference field holograms
        """
        self.params = params
        self.target = target
        self.reference = reference

    def get_field_and_spectrum(
        self,
        offset_regs: list[tuple[int, int], tuple[int, int]] | None = None,
        crop_center: bool = False,
        c_r: int = 5,
    ):
        """get the field and spectrum from the hologram arrays

        Args:
            offset_regs (list[tuple[int, int], tuple[int, int]] | None, optional): offset regions. Defaults to None.
            crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.
        """
        print("convert to field...")
        self.field = []
        self.spectrum = []
        for i in tqdm(range(len(self.target))):
            array = self.target[i]
            ref_array = self.reference[i]

            array_field, array_field_fft = preprocess_for_synthesis(
                array, ref_array, self.params, offset_regs, crop_center, c_r
            )

            self.field.append(array_field)
            self.spectrum.append(array_field_fft)

    def save_multiangle_qpi(self, path: str = "multiangle_qpi"):
        """save multiangle QPI images

        Args:
            path (str, optional): Path to save QPIs. Defaults to "multiangle_qpi".
        """
        assert len(self.field) != 0
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("saving...")
        for i in tqdm(range(len(self.field))):
            if _cp:
                to_save = xp.asnumpy(xp.angle(self.field[i]))
            else:
                to_save = xp.angle(self.field[i])
            plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")

    def save_multiangle_spectrum(self, path: str = "multiangle_spectrum"):
        """save multiangle spectrum images

        Args:
            path (str, optional): Path to save spectrum images. Defaults to "multiangle_spectrum".
        """
        assert len(self.spectrum) != 0
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("saving...")
        for i in tqdm(range(len(self.spectrum))):
            if _cp:
                to_save = xp.asnumpy(xp.log(xp.abs(self.spectrum[i])))
            else:
                to_save = xp.log(xp.abs(self.spectrum[i]))
            plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")

    def synthesize_spectrums(self) -> tuple[xp.array, xp.array]:
        """synthesize the spectrums

        Returns:
            tuple[xp.array, xp.array]: synthesized array and its Fourier transform
        """
        assert len(self.field) != 0
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
            ),
            dtype=xp.complex128,
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        print("synthesizing...")
        for i in tqdm(range(len(self.target))):
            fft_field = self.spectrum[i]

            synthesized_fft += fft_field
            synthesized_weight += fft_field != 0

        synthesized_fft /= synthesized_weight
        synthesized_array = xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft))

        return synthesized_array, synthesized_fft

    def qpi(self, offset_regs: list[tuple[int, int], tuple[int, int]] | None = None) -> tuple[xp.array, xp.array]:
        """Quantitative phase imaging (QPI) calculation

        Args:
            offset_regs (list[tuple[int, int], tuple[int, int]] | None, optional): offset regions. Defaults to None.

        Returns:
            tuple[xp.array, xp.array]: synthesized QPI and its Fourier transform
        """
        self.get_field_and_spectrum(offset_regs=offset_regs)
        synthesized_array, synthesized_fft = self.synthesize_spectrums()
        synthesized_qpi = xp.angle(synthesized_array)

        return synthesized_qpi, synthesized_fft

    def mipqpi(
        self, offset_regs: list[tuple[int, int], tuple[int, int]] | None = None, c_r: int = 5
    ) -> tuple[xp.array, xp.array]:
        """Mid-infrared Photothermal Quantitative Phase imaging (MIPQPI) calculation

        Args:
            offset_regs (list[tuple[int, int], tuple[int, int]] | None, optional): offset regions. Defaults to None.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

        Returns:
            tuple[xp.array, xp.array]: synthesized MIPQPI and its Fourier transform
        """
        self.get_field_and_spectrum(offset_regs=offset_regs, crop_center=True, c_r=c_r)
        synthesized_array, synthesized_fft = self.synthesize_spectrums()
        synthesized_mipqpi = xp.angle(synthesized_array)

        return synthesized_mipqpi, synthesized_fft
