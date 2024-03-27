from __future__ import annotations

import os
import uuid
from typing import NewType, Union

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

from muscopy.cfg import EDGE_SIZE, Regions
from muscopy.qpi import QPIParameters, correct_offset, make_disk


class ODTParameters(QPIParameters):
    def __init__(
        self,
        wavelength: float,
        NA: float,
        img_shape: tuple[int, int],
        pixelsize: float,
        offaxis_center: tuple[int, int],
        n_sol: float = 1.33,
        NA_illumi: float | None = None,
    ):
        super().__init__(wavelength, NA, img_shape, pixelsize, offaxis_center)
        self.n_sol = n_sol
        self.NA_illumi = NA_illumi

        self._calc_params()

    def _calc_params(self):
        super()._calc_params()
        self.fi_mag = self.n_sol / self.wav / self.freq_per_pixel  # |k| in terms of pixel unit

        self.k_per_pixel = 2 * np.pi * self.freq_per_pixel  # unit of k in terms of pixel unit
        self.imgpx_unit = (
            self.pixelsize * self.img_shape[0] / (2 * self.aperturesize + 1)
        )  # unit image pixel size on the cropped image plane

        if self.NA_illumi is not None:
            self.fi_lateral_mag = self.fi_mag * self.NA_illumi / self.n_sol  # |k_T| in terms of pixel unit

            self.fi_z = int(
                self.fi_mag * (1 - self.NA_illumi**2 / self.n_sol**2) ** 0.5
            )  # |k_z| in terms of pixel unit

            self.fz_extent = int((self.fi_mag - self.fi_z))  # extent of kz axis in terms of pixel unit
            self.imgpx_unit_z = self.imgpx_unit * (
                (2 * self.aperturesize + 1) / (2 * self.fz_extent + 1)
            )  # unit image pixel size along z axis on the cropped image plane


Params = Union[QPIParameters, ODTParameters]


def find_max_args(array: xp.array) -> tuple[int, int, float]:
    """Find maximum value and its index in the array

    Args:
        array (xp.array): array to find maximum value

    Returns:
        tuple[int, int, float]: position of max value, and max value itself
    """
    max_value = xp.max(array)
    max_idx = xp.unravel_index(xp.argmax(array), array.shape)
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


def get_oblique_field(array: xp.array, params: Params) -> tuple[xp.array, tuple[int, int]]:
    """internal method. get the electric field from the hologram array

    Args:
        array (xp.array): input array
        params (Params): Parameters class

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
    array_field: xp.array,
    ref_array_field: xp.array,
    oblique_shift: tuple[int, int],
    params: Params,
    offset_regs: Regions | None = None,
    crop_center: bool = False,
    c_r: int = 5,
) -> tuple[xp.array, xp.array]:
    """internal method. preprocess the array for synthesis

    Args:
        array_field (xp.array): sample complex field
        ref_array_field (xp.array): reference complex field
        params (Params): Parameters class
        oblique_shift (tuple[int, int]): oblique shift in the Fourier space
        offset_regs (Regions optional): offset regions. Defaults to None.
        crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
        c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

    Returns:
        tuple[xp.array, xp.array]: preprocessed array and its Fourier transform
    """
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


def get_scattering_field(
    array_field: xp.array,
    ref_array_field: xp.array,
    oblique_shift: tuple[int, int],
    params: Params,
    approx: str,
    crop_center: bool = False,
    c_r: int = 5,
) -> tuple[xp.array, xp.array]:
    assert approx in ["Born", "Rytov"]
    if approx == "Born":
        scattering = (array_field - ref_array_field) / ref_array_field
    elif approx == "Rytov":
        log_array = xp.log(array_field)
        log_ref_array = xp.log(ref_array_field)
        scattering = log_array - log_ref_array
    else:
        raise ValueError("approx should be either 'Born' or 'Rytov'")

    scattering_fft = xp.fft.fftshift(xp.fft.fft2(scattering))
    disk_for_synthesis = make_disk(
        (
            params.aperturesize - oblique_shift[0],
            params.aperturesize - oblique_shift[1],
        ),
        params.aperturesize // 2,
        scattering.shape,
    )
    scattering_fft = scattering_fft * disk_for_synthesis

    if crop_center:
        mask_highpass = make_disk(
            (params.aperturesize - oblique_shift[0], params.aperturesize - oblique_shift[1]),
            c_r,
            scattering.shape,
            highpass=True,
        )
        scattering_fft = scattering_fft * mask_highpass

    return scattering, scattering_fft


############################################
# Synthetic-Aperture (MIP)QPI and (MIP)ODT #
############################################


class DataHolder:
    def __init__(self, identifier: str | None = None, **kwargs):
        if identifier is None:
            self.identifier = str(uuid.uuid4())
        else:
            self.identifier = identifier

        self.sample_field: xp.array = None
        self.reference_field: xp.array = None
        self.oblique_shift: tuple[int, int] = None

        self.div_field: xp.array = None
        self.spectrum: xp.array = None

        self.scattering: xp.arraye = None
        self.scattering_spectrum: xp.array = None

        self.tags = kwargs

    def get_identifier(self) -> str:
        return self.identifier


class Synthesizer:
    def __init__(self, params: Params):
        """Synthesizer class for aperture synthesis

        Args:
            params (Params): Parameters class
        """
        self.params = params

        self.identifiers: list[str] = list()
        self.data: dict[str, DataHolder] = dict()

    def set_sample_data(self, sample_data: list[xp.array]):
        """set sample data

        Args:
            sample_data (list[xp.array]): sample holograms
        """
        self.sample = sample_data

    def set_reference_data(self, reference_data: list[xp.array]):
        """set reference data

        Args:
            reference_data (list[xp.array]): reference holograms
        """
        self.reference = reference_data

    def get_field(self):
        """get the field from the hologram arrays"""
        print("get field...")
        for i in tqdm(range(len(self.sample))):
            data = DataHolder()
            id = data.get_identifier()
            self.identifiers.append(id)
            self.data[id] = data

            array = self.sample[i]
            ref_array = self.reference[i]

            array_field, oblique_shift = get_oblique_field(array, self.params)
            ref_array_field, _ = get_oblique_field(ref_array, self.params)

            data.sample_field = array_field
            data.reference_field = ref_array_field
            data.oblique_shift = oblique_shift

    def get_div_field_and_spectrum(
        self,
        offset_regs: Regions | None = None,
        crop_center: bool = False,
        c_r: int = 5,
    ):
        """get the divided field and its Fourier transform

        Args:
            offset_regs (Regions | None, optional): offset regions. Defaults to None.
            crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.
        """
        print("preprocessing for Aperture Synthesis...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            array_div, array_div_fft = preprocess_for_synthesis(
                data.sample_field,
                data.reference_field,
                data.oblique_shift,
                self.params,
                offset_regs=offset_regs,
                crop_center=crop_center,
                c_r=c_r,
            )

            data.div_field = array_div
            data.spectrum = array_div_fft

    def get_scattering_field(
        self, approx: str, offset_regs: Regions | None = None, crop_center: bool = False, c_r: int = 5
    ):
        """get the scattering field

        Args:
            approx (str): approximation for the ODT calculation
            offset_regs (Regions | None, optional): offset regions. Defaults to None.
            crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.
        """
        print("preprocessing for ODT...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            (scattering, scattering_fft) = get_scattering_field(
                data.sample_field,
                data.reference_field,
                data.oblique_shift,
                self.params,
                approx,
                crop_center=crop_center,
                c_r=c_r,
            )

            data.scattering = scattering
            data.scattering_spectrum = scattering_fft

    def save_multiangle_qpi(self, path: str = "multiangle_qpi"):
        """save multiangle QPI images

        Args:
            path (str, optional): Path to save QPIs. Defaults to "multiangle_qpi".
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("saving...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            if _cp:
                to_save = xp.asnumpy(xp.angle(data.div_field))
            else:
                to_save = xp.angle(data.div_field)
            plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")

    def save_multiangle_spectrum(self, path: str = "multiangle_spectrum"):
        """save multiangle spectrum images

        Args:
            path (str, optional): Path to save spectrum images. Defaults to "multiangle_spectrum".
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("saving...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            if _cp:
                to_save = xp.asnumpy(xp.log(xp.abs(data.spectrum)))
            else:
                to_save = xp.log(xp.abs(data.spectrum))
            plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")

    def synthesize_spectrums(self) -> tuple[xp.array, xp.array]:
        """synthesize the spectrums

        Returns:
            tuple[xp.array, xp.array]: synthesized array and its Fourier transform
        """
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
            ),
            dtype=xp.complex128,
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        print("synthesizing...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            fft_field = data.spectrum[i]

            synthesized_fft += fft_field
            synthesized_weight += fft_field != 0

        synthesized_fft /= synthesized_weight
        synthesized_array = xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft))

        return synthesized_array, synthesized_fft

    def synthesize_on_3d(self, hermite: bool = False) -> tuple[xp.array, xp.array]:
        """synthesize the spectrums on 3D

        Args:
            hermite (bool, optional): whether to use Hermite symmetry. Defaults to False.

        Returns:
            tuple[xp.array, xp.array]: synthesized array and its Fourier transform
        """
        assert isinstance(self.params, ODTParameters)
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - EDGE_SIZE,
                self.params.fz_extent * 2 + 1,
            ),
            dtype=xp.complex128,
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        print("ODT Synthesizing...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            fft_field = data.scattering_spectrum
            kz_disk = calc_kz_value(self.params, data.oblique_shift)

            scatter_potential_fft = 2j * kz_disk * fft_field

            scatter_potential_fft3d = map_aperture_to_Ewald(
                scatter_potential_fft,
                synthesized_fft.shape,
                data.oblique_shift,
                self.params,
            )

            synthesized_fft += scatter_potential_fft3d
            synthesized_weight += scatter_potential_fft3d != 0

            if hermite:
                conj_scatter_potential_fft_3d = xp.conjugate(xp.flip(scatter_potential_fft3d, axis=(0, 1, 2)))

                synthesized_fft += conj_scatter_potential_fft_3d
                synthesized_weight += conj_scatter_potential_fft_3d != 0

        synthesized_weight -= synthesized_weight == 1
        synthesized_fft /= synthesized_weight

        synthesized_array = xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft))

        synthesized_array = xp.fft.fftshift(synthesized_array, axes=(2))
        return synthesized_array, synthesized_fft

    def iterative_reconstruct(
        self,
        array3d: xp.array,
        array3dfft: xp.array,
        scale_diff: float = 1,
        n_iter: int = 100,
        epsilon: float = 0.1,
        interval: int = 100,
    ) -> xp.array:
        """Iterative reconstruction

        Args:
            array3d (xp.array): 3D array to be reconstructed
            array3dfft (xp.array): Fourier transform of the 3D array
            scale_diff (float, optional): scale difference between spatial array and spectral array. Defaults to 1.
            n_iter (int, optional): maximum number of iterations. Defaults to 100.
            epsilon (float, optional): epsilon for the convergence. Defaults to 0.1.
            interval (int, optional): interval to print the error. Defaults to 100.

        Returns:
            xp.array: reconstructed 3D array
        """
        print("Iterative reconstruction...")
        tmp_array = array3d.copy()
        last_array = array3d.copy()
        err = np.inf
        count = 0
        while (err > epsilon) and (count < n_iter):
            tmp_array[xp.real(tmp_array) < 0] = 0
            tmp_fft = xp.fft.fftshift(xp.fft.fftn(tmp_array)) * scale_diff
            tmp_fft[array3dfft != 0] = array3dfft[array3dfft != 0]
            tmp_array = xp.fft.ifftn(xp.fft.ifftshift(tmp_fft)) / scale_diff

            err = calc_normalized_L2error(tmp_array, last_array)
            last_array = tmp_array.copy()

            count += 1
            if count % interval == 0:
                err = calc_normalized_L2error(array3d, last_array)
                print(f"iter: {count}, error: {err}")
                last_array = array3d.copy()

        return last_array

    def qpi(self, offset_regs: Regions | None = None) -> tuple[xp.array, xp.array]:
        """Quantitative phase imaging (QPI) calculation

        Args:
            offset_regs (Regions | None, optional): offset regions. Defaults to None.

        Returns:
            tuple[xp.array, xp.array]: synthesized QPI and its Fourier transform
        """
        self.get_field()
        self.get_div_field_and_spectrum(offset_regs=offset_regs)
        synthesized_array, synthesized_fft = self.synthesize_spectrums()
        synthesized_qpi = xp.angle(synthesized_array)

        return synthesized_qpi, synthesized_fft

    def mipqpi(self, offset_regs: Regions | None = None, c_r: int = 5) -> tuple[xp.array, xp.array]:
        """Mid-infrared Photothermal Quantitative Phase imaging (MIPQPI) calculation

        Args:
            offset_regs (Regions | None, optional): offset regions. Defaults to None.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

        Returns:
            tuple[xp.array, xp.array]: synthesized MIPQPI and its Fourier transform
        """
        self.get_field()
        self.get_div_field_and_spectrum(offset_regs=offset_regs, crop_center=True, c_r=c_r)
        synthesized_array, synthesized_fft = self.synthesize_spectrums()
        synthesized_mipqpi = xp.angle(synthesized_array)

        return synthesized_mipqpi, synthesized_fft

    def odt(self):
        pass

    def mipodt(self):
        pass


####################################################
# Methods for Optical Diffraction Tomography (ODT) #
####################################################


def map_aperture_to_Ewald(
    array: xp.array,
    shape: tuple[int, int, int],
    oblique_shift: tuple[int, int],
    params: ODTParameters,
) -> xp.array:
    """map 2d array to 3d array(ODT)

    Args:
        array (xp.array): 2D array to be projected. array size should be 2*aperturesize+1 square.
        shape (tuple[int, int, int]): shape of the output 3d array. xy shape must be consistent with the input array
        oblique_shift (tuple): oblique shift in the Fourier space
        params (ODTParameters): parameters for the ODT system

    Returns:
        xp.array: 3D array projected to the Ewald sphere
    """
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
    circle = (xx - params.aperturesize + oblique_shift[0]) ** 2 + (yy - params.aperturesize + oblique_shift[1]) ** 2
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


def calc_kz_value(
    params: ODTParameters,
    oblique_shift: tuple[int, int],
) -> xp.array:
    xx, yy = xp.meshgrid(
        xp.arange(2 * params.aperturesize + 1),
        xp.arange(2 * params.aperturesize + 1),
        indexing="ij",
    )
    disk = (xx - params.aperturesize + oblique_shift[0]) ** 2 + (yy - params.aperturesize + oblique_shift[1]) ** 2
    disk_mask = disk < (params.aperturesize // 2) ** 2
    fz_disk = (params.fi_mag**2 - disk) * disk_mask
    fz_disk = fz_disk**0.5
    kz_disk = fz_disk * params.k_per_pixel

    return kz_disk


def calc_refractive_index_square(array3d: xp.array, params: ODTParameters) -> xp.array:
    """calculate :math:`n^2` for the ODT calculation

    Args:
        array3d (xp.array): Array to calculate refractive index square
        params (ODTParameters): parameters for the ODT system

    Returns:
        xp.array: complex refractive index
    """
    r_3d_square = params.n_sol**2 * (
        xp.ones(array3d.shape, dtype=xp.complex128) - array3d / (params.fi_mag * params.k_per_pixel) ** 2
    )
    return r_3d_square


def discard_z(array: xp.array, threshold: int) -> xp.array:
    """internal method for `discard_higher_kz`. discard the higher z values

    Args:
        array (xp.array): original array(usually Fourier transformed array)
        threshold (int): cut off threshold

    Returns:
        xp.array: lowpassed array
    """
    array = array[:, :, threshold : array.shape[2] - threshold]
    return array


def discard_higher_kz(array: xp.array, threshold: int) -> xp.array:
    """discard the higher z values

    Args:
        array (xp.array): original array
        threshold (int): cut off threshold

    Returns:
        xp.array: lowpassed array
    """
    norm_factor = array.shape[2]
    array_fft = xp.fft.fftshift(xp.fft.fftn(array, norm="backward"))
    discarded = discard_z(array_fft, threshold)
    new_array = xp.fft.ifftn(xp.fft.ifftshift(discarded), norm="forward") / norm_factor**3
    return new_array


def zeropad_higher_kz(array: xp.array, extend: int) -> xp.array:
    """zero pad the higher z values

    Args:
        array (xp.array): original array
        extend (int): extend size

    Returns:
        xp.array: zero padded array
    """
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
    new_array = xp.fft.ifftn(xp.fft.ifftshift(new_array_fft), norm="forward") / norm_factor
    return new_array


def calc_full_volume(params: ODTParameters) -> xp.float:
    """Calculate the volume which can be filled with FW light under the given parameters

    Args:
        params (ODTParameters): parameters for the ODT system

    Returns:
        xp.float : Volume
    """
    theta = xp.arcsin(params.aperturesize / (2 * params.fi_mag))
    S = 2 * theta * params.fi_mag**2 - params.aperturesize * params.fi_mag * xp.cos(theta)
    V = S * xp.pi * params.aperturesize
    return V


def calc_normalized_L2error(array: xp.array, ref_array: xp.array) -> float:
    """Calculate the normalized L2 error

    Args:
        array (xp.array): array to be compared
        ref_array (xp.array): reference array

    Returns:
        float: normalized L2 error
    """
    error = xp.sum(xp.abs(array - ref_array)) / xp.sum(xp.abs(ref_array))
    return error
