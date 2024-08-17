from __future__ import annotations

import os
import pickle
import shutil
import uuid
from dataclasses import dataclass
from functools import cached_property
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import muscopy.cfg as mcfg
from muscopy.cfg import ArrayPrecision, OffsetRegions
from muscopy.qpi import QPIParameters, correct_offset, make_disk
from muscopy.unwrap_phase import phase_unwrap

if mcfg._cp:
    import cupy as xp
    import cupy as cp
else:
    import numpy as xp


@dataclass
class ODTParameters(QPIParameters):
    NA_illumi: float = 0
    zmargin: int = 3

    @cached_property
    def fi_lateral_mag(self) -> float:
        return self.fi_mag * self.NA_illumi / self.n_sol

    @cached_property
    def fi_z(self) -> int:
        return int(self.fi_mag * (1 - self.NA_illumi**2 / self.n_sol**2) ** 0.5)

    @cached_property
    def fz_extent(self) -> int:
        fz_extent_top = int(self.fi_mag - self.fi_z)
        fz_extent_buttom = int(self.fi_mag * (self.n_sol - (self.n_sol**2 - self.NA**2) ** 0.5))
        return max(fz_extent_top, fz_extent_buttom) + self.zmargin

    @cached_property
    def imgpx_unit_z(self) -> float:
        return self.imgpx_unit * ((2 * self.aperturesize + 1) / (2 * self.fz_extent + 1 + 6))

    @cached_property
    def S2Fz(self) -> float:
        return (self.imgpx_unit_z / self.k_per_pixel) ** 0.5

    @cached_property
    def F2Sz(self) -> float:
        return (self.k_per_pixel / self.imgpx_unit_z) ** 0.5

    @cached_property
    def fi_lateral_mag(self) -> float:
        return self.fi_mag * self.NA_illumi / self.n_sol

    @cached_property
    def fi_z(self) -> int:
        return int(self.fi_mag * (1 - self.NA_illumi**2 / self.n_sol**2) ** 0.5)

    @cached_property
    def fz_extent(self) -> int:
        fz_extent_top = int(self.fi_mag - self.fi_z)
        fz_extent_buttom = int(self.fi_mag * (self.n_sol - (self.n_sol**2 - self.NA**2) ** 0.5))
        return max(fz_extent_top, fz_extent_buttom) + self.zmargin

    @cached_property
    def imgpx_unit_z(self) -> float:
        return self.imgpx_unit * ((2 * self.aperturesize + 1) / (2 * self.fz_extent + 1 + 6))

    @cached_property
    def S2Fz(self) -> float:
        return (self.imgpx_unit_z / self.k_per_pixel) ** 0.5

    @cached_property
    def F2Sz(self) -> float:
        return (self.k_per_pixel / self.imgpx_unit_z) ** 0.5

    def print_all_parameters(self):
        self._print_all_parameters()
        # calculated parameters
        print(f"fi_lateral_mag: {self.fi_lateral_mag}")
        print(f"fi_z: {self.fi_z}")
        print(f"fz_extent: {self.fz_extent}")
        print(f"imgpx_unit_z: {self.imgpx_unit_z}")
        print(f"S2Fz: {self.S2Fz}")
        print(f"F2Sz: {self.F2Sz}")


Params = Union[QPIParameters, ODTParameters]


def find_max_args(array: NDArray) -> tuple[int, int, float]:
    """Find maximum value and its index in the array

    Args:
        array (NDArray): array to find maximum value

    Returns:
        tuple[int, int, float]: position of max value, and max value itself
    """
    max_value = xp.max(array)
    max_idx = xp.unravel_index(xp.argmax(array), array.shape)
    max_x = max_idx[0]
    max_y = max_idx[1]

    return max_x, max_y, max_value


def crop_oblique_array(
    array: NDArray, oblique_center: tuple[int, int], aperturesize: int
) -> tuple[NDArray, tuple[int, int]]:
    """Crop the array with the oblique center and aperturesize

    Args:
        array (NDArray): input array
        oblique_center (tuple[int, int]): center of the oblique array
        aperturesize (int): size of the aperturesize

    Returns:
        NDArray: cropped array
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


def get_oblique_spectrum(array: NDArray, params: Params) -> tuple[NDArray, tuple[int, int]]:
    """internal method. get the spectrum from the hologram array

    Args:
        array (NDArray): input array
        params (Params): Parameters class

    Returns:
        NDArray: spectrum from the hologram array
        tuple[int, int]: oblique shift in the Fourier space
    """
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    mask = make_disk(params.offaxis_center, params.aperturesize / 2, params.img_shape)
    array_fft = array_fft * mask

    array_fft, oblique_shift = crop_oblique_array(array_fft, params.offaxis_center, params.aperturesize)

    return array_fft, oblique_shift


def get_oblique_field(array: NDArray, params: Params) -> tuple[NDArray, tuple[int, int]]:
    """internal method. get the electric field from the hologram array

    Args:
        array (NDArray): input array
        params (Params): Parameters class

    Returns:
        NDArray: field from the hologram array
        tuple[int, int]: oblique shift in the Fourier space
    """
    array_fft, oblique_shift = get_oblique_spectrum(array, params)
    # correct the scale factor caused by the cropping
    scale_factor = len(array_fft) / len(array)
    array_fft = array_fft * scale_factor
    # remove EDGE to avoid the edge effect
    array_field = xp.fft.ifft2(xp.fft.ifftshift(array_fft))[mcfg.EDGE_SIZE :, mcfg.EDGE_SIZE :] * params.F2S**2

    return array_field, oblique_shift


def preprocess_for_synthesis(
    array_field: NDArray,
    ref_array_field: NDArray,
    oblique_shift: tuple[int, int],
    params: Params,
    MIP: bool = False,
    crop_center: bool = False,
    c_r: int = 5,
) -> tuple[NDArray, NDArray]:
    """internal method. preprocess the array for synthesis

    Args:
        array_field (NDArray): sample complex field
        ref_array_field (NDArray): reference complex field
        params (Params): Parameters class
        oblique_shift (tuple[int, int]): oblique shift in the Fourier space
        MIP (bool, optional): whether to correct on/off of MIR pump. Defaults to False.
        crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
        c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

    Returns:
        tuple[NDArray, NDArray]: preprocessed array and its Fourier transform
    """
    array_div = array_field / ref_array_field

    if mcfg.OFFSET_REGS is not None:
        array_div = correct_offset(array_div, mcfg.OFFSET_REGS)

    if MIP and (mcfg.MIPRegion is not None):
        center_phase = xp.mean(
            xp.angle(
                array_div[mcfg.MIP_CENTER[0][0] : mcfg.MIP_CENTER[0][1], mcfg.MIP_CENTER[1][0] : mcfg.MIP_CENTER[1][1]]
            )
        )
        # if center_phase < 0:
        #     array_div = 1 / array_div

    array_div_fft = xp.fft.fftshift(xp.fft.fft2(array_div)) * params.S2F**2
    center_x = params.aperturesize - mcfg.EDGE_SIZE // 2
    center_y = params.aperturesize - mcfg.EDGE_SIZE // 2
    disk_for_synthesis = make_disk(
        (
            center_x - oblique_shift[0],
            center_y - oblique_shift[1],
        ),
        params.aperturesize // 2,
        array_field.shape,
    )

    array_div_fft = array_div_fft * disk_for_synthesis

    # for mipqpi
    if crop_center:
        mask_highpass = make_disk(
            (center_x - oblique_shift[0], center_y - oblique_shift[1]),
            c_r,
            array_div.shape,
            highpass=True,
        )
        array_div_fft = array_div_fft * mask_highpass

    return array_div, array_div_fft


def get_scattering_field(
    array_field: NDArray,
    ref_array_field: NDArray,
    oblique_shift: tuple[int, int],
    params: Params,
    approx: str = "Rytov",
    MIP: bool = False,
    crop_center: bool = False,
    c_r: int = 5,
) -> tuple[NDArray, NDArray]:
    """internal method. get the scattering field

    Args:
        array_field (NDArray): sample complex field
        ref_array_field (NDArray): reference complex field
        oblique_shift (tuple[int, int]): oblique shift in the Fourier space
        params (Params): Parameters class
        approx (str, optional): approximation for the ODT calculation. Defaults to "Rytov".
        MIP (bool, optional): whether to correct on/off of MIR pump. Defaults to False.
        crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
        c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.

    Returns:
        tuple[NDArray, NDArray]: scattering field and its Fourier transform
    """
    assert approx in ["Born", "Rytov"]

    array_field = correct_offset(array_field, mcfg.OFFSET_REGS)
    ref_array_field = correct_offset(ref_array_field, mcfg.OFFSET_REGS)

    if MIP and (mcfg.MIPRegion is not None):
        array_div = array_field / ref_array_field
        center_phase = xp.mean(
            xp.angle(
                array_div[mcfg.MIP_CENTER[0][0] : mcfg.MIP_CENTER[0][1], mcfg.MIP_CENTER[1][0] : mcfg.MIP_CENTER[1][1]]
            )
        )
        # if center_phase < 0:
        #     array_field, ref_array_field = ref_array_field, array_field

    if approx == "Born":
        scattering = (array_field - ref_array_field) / ref_array_field
    elif approx == "Rytov":
        array_log = xp.log(array_field)
        array_log_real = xp.real(array_log)
        array_log_imag = xp.imag(array_log)
        ref_array_log = xp.log(ref_array_field)
        ref_array_log_real = xp.real(ref_array_log)
        ref_array_log_imag = xp.imag(ref_array_log)
        # array_log_imag_unwrap = phase_unwrap(array_log_imag)
        # ref_array_log_imag_unwrap = phase_unwrap(ref_array_log_imag)

        amplitude = array_log_real - ref_array_log_real
        # phase = array_log_imag_unwrap - ref_array_log_imag_unwrap
        phase = phase_unwrap(array_log_imag - ref_array_log_imag)

        if mcfg.OFFSET_REGS is not None:
            amplitude = correct_amplitude_offset(amplitude, mcfg.OFFSET_REGS)
            phase = correct_phase_offset(phase, mcfg.OFFSET_REGS)

        scattering = amplitude + 1j * phase

    else:
        raise ValueError("approx should be either 'Born' or 'Rytov'")

    scattering_fft = (
        xp.fft.fftshift(xp.fft.fft2(scattering, norm="ortho")) * params.S2F**2 * (2 * xp.pi)
    )  # last factor is to adjust to the non-Unitary derivation in Tamamitsu's paper
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


def correct_phase_offset(phase: NDArray, offset_regs: OffsetRegions) -> NDArray:
    """internal method. correct phase offset for the scattering field

    Args:
        phase (NDArray): input phase array
        offset_regs (OffsetRegions): regions for offset calculation

    Returns:
        NDArray: corrected phase array
    """
    if offset_regs is None:
        return phase
    phase_offset_list = []
    for region in offset_regs:
        phase_offset_list.append(xp.mean(phase[region[0][0] : region[0][1], region[1][0] : region[1][1]]))
    phase_offset = xp.mean(xp.array(phase_offset_list))

    phase = phase - phase_offset

    return phase


def correct_amplitude_offset(amplitude: NDArray, offset_regs: OffsetRegions) -> NDArray:
    """internal method. correct amplitude offset for the scattering field

    Args:
        amplitude (NDArray): input complex array
        offset_regs (OffsetRegions): regions for offset calculation

    Returns:
        NDArray: corrected array
    """
    if offset_regs is None:
        return amplitude
    amplitude_offset_list = []
    for region in offset_regs:
        amplitude_offset_list.append(xp.mean(amplitude[region[0][0] : region[0][1], region[1][0] : region[1][1]]))
    amplitude_offset = xp.mean(xp.array(amplitude_offset_list))

    amplitude = amplitude - amplitude_offset

    return amplitude


############################################
# Synthetic-Aperture (MIP)QPI and (MIP)ODT #
############################################


class DataHolder:
    def __init__(self, identifier: str | None = None, **kwargs):
        if identifier is None:
            self.identifier = str(uuid.uuid4())
        else:
            self.identifier = identifier

        self.sample_field: NDArray | None = None  # TODO: resolve mypy error
        self.sample_spectrum: NDArray | None = None
        self.reference_field: NDArray | None = None
        self.reference_spectrum: NDArray | None = None
        self.oblique_shift: tuple[int, int] | None = None

        self.div_field: NDArray | None = None
        self.div_spectrum: NDArray | None = None

        self.scattering: NDArray | None = None
        self.scattering_spectrum: NDArray | None = None

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

    def initialize_data(self):
        """initialize the data"""
        self.data = dict()
        self.identifiers = list()

    def set_sample_data(self, sample_data: list[NDArray]):
        """set sample data

        Args:
            sample_data (list[NDArray]): sample holograms
        """
        self.sample = sample_data

    def set_reference_data(self, reference_data: list[NDArray]):
        """set reference data

        Args:
            reference_data (list[NDArray]): reference holograms
        """
        self.reference = reference_data

    def set_sample_data_from_path(self, path: list[str]):
        """set sample data from the path

        Args:
            path (str): path to the sample holograms
        """
        path.sort()
        self.sample = [xp.load(path) for path in path]

    def set_reference_data_from_path(self, path: list[str]):
        """set reference data from the path

        Args:
            path (str): path to the reference holograms
        """
        path.sort()
        self.reference = [xp.load(path) for path in path]

    def set_sample_compressed_from_path(self, path: list[str]):
        """set sample data from the path

        Args:
            path (str): path to the pickle file of the compressed sample holograms
        """
        path.sort()
        self.sample_dh = [pickle.load(open(pkl_path, "rb")) for pkl_path in path]

    def set_reference_compressed_from_path(self, path: list[str]):
        """set reference data from the path

        Args:
            path (str): path to the pickle file of the compressed reference holograms
        """
        path.sort()
        self.reference_dh = [pickle.load(open(pkl_path, "rb")) for pkl_path in path]

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

    def get_field_from_compressed(self):
        """get the field from the compressed hologram arrays"""
        print("get field...")
        for i in tqdm(range(len(self.sample_dh))):
            data = self.sample_dh[i]
            ref_data = self.reference_dh[i]
            id = data.get_identifier()
            self.identifiers.append(id)
            self.data[id] = data

            data.oblique_shift = ref_data.oblique_shift
            data.reference_field = ref_data.sample_field

    def get_div_field_and_spectrum(
        self,
        MIP: bool = False,
        crop_center: bool = False,
        c_r: int = 5,
    ):
        """get the divided field and its Fourier transform

        Args:
            MIP (bool, optional): whether to correct on/off of MIR pump. Defaults to False.
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
                MIP=MIP,
                crop_center=crop_center,
                c_r=c_r,
            )

            data.div_field = array_div
            data.div_spectrum = array_div_fft

    def get_scattering_field(self, approx: str = "Rytov", MIP: bool = False, crop_center: bool = False, c_r: int = 5):
        """get the scattering field

        Args:
            approx (str): approximation for the ODT calculation
            MIP (bool, optional): whether to correct on/off of MIR pump. Defaults to False.
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
                MIP=MIP,
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
            qpi = xp.angle(data.div_field)
            qpi_unwrap = phase_unwrap(qpi)
            if mcfg._cp:
                to_save = xp.asnumpy(qpi_unwrap)
            else:
                to_save = xp.angle(qpi_unwrap)
            # plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")
            fig, ax = plt.subplots()
            cax = ax.imshow(to_save, vmin=0)
            fig.colorbar(cax, ax=ax)
            plt.savefig(f"{path}/{i:03}.png")
            plt.close(fig)

    def save_scattering_field(self, path: str = "multiangle_scattering"):
        """save scattering field images

        Args:
            path (str, optional): Path to save scattering field images. Defaults to "scattering_field".
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        print("saving...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            if mcfg._cp:
                to_save = xp.asnumpy(xp.abs(data.scattering))
            else:
                to_save = xp.imag(data.scattering)
            # plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")
            fig, ax = plt.subplots()
            cax = ax.imshow(to_save, vmin=0)
            fig.colorbar(cax, ax=ax)
            plt.savefig(f"{path}/{i:03}.png")
            plt.close(fig)

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
            if mcfg._cp:
                to_save = xp.asnumpy(xp.log(xp.abs(data.div_spectrum) + 1e-60))
            else:
                to_save = xp.log(xp.abs(data.div_spectrum))
            plt.imsave(f"{path}/{i:03}.png", to_save, cmap="gray")

    def synthesize_spectrums(self, precision: ArrayPrecision = None) -> tuple[NDArray, NDArray]:
        """synthesize the spectrums

        Returns:
            tuple[NDArray, NDArray]: synthesized array and its Fourier transform
        """
        if precision is None:
            precision = ArrayPrecision(32, 64)
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - mcfg.EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - mcfg.EDGE_SIZE,
            ),
            dtype=precision.get_complex_precision(),
        )
        synthesized_weight = xp.ones(synthesized_fft.shape)

        print("synthesizing...")
        for i in tqdm(range(len(self.identifiers))):
            data = self.data[self.identifiers[i]]
            fft_field = data.div_spectrum

            synthesized_fft += fft_field
            synthesized_weight += fft_field != 0

        synthesized_weight -= synthesized_weight > 1
        synthesized_fft /= synthesized_weight
        synthesized_array = xp.fft.ifft2(xp.fft.ifftshift(synthesized_fft)) * self.params.F2S**2

        return synthesized_array, synthesized_fft

    def synthesize_on_3d(
        self, hermite: bool = False, precision: ArrayPrecision | None = None
    ) -> tuple[NDArray, NDArray]:
        """synthesize the spectrums on 3D

        Args:
            hermite (bool, optional): whether to use Hermite symmetry. Defaults to False.
            precision (ArrayPrecision | None, optional): precision of the array. Defaults to None.

        Returns:
            tuple[NDArray, NDArray]: synthesized array and its Fourier transform
        """
        assert isinstance(self.params, ODTParameters)
        if precision is None:
            precision = ArrayPrecision(32, 64)
        synthesized_fft = xp.zeros(
            (
                2 * (self.params.aperturesize) + 1 - mcfg.EDGE_SIZE,
                2 * (self.params.aperturesize) + 1 - mcfg.EDGE_SIZE,
                self.params.fz_extent * 2 + 1,
            ),
            dtype=precision.get_complex_precision(),
        )
        synthesized_weight = xp.ones(synthesized_fft.shape, dtype=precision.get_int_precision())

        print("ODT Synthesizing...")
        for i in tqdm(range(0, len(self.identifiers), 1)):
            data = self.data[self.identifiers[i]]
            fft_field = data.scattering_spectrum.astype(precision.get_complex_precision())
            kz_disk = calc_kz_value(self.params, data.oblique_shift, precision)

            scatter_potential_fft = 2j * kz_disk * fft_field

            scatter_potential_fft3d = map_aperture_to_Ewald(
                scatter_potential_fft, synthesized_fft.shape, data.oblique_shift, self.params, precision
            )

            # TODO: make option to transfer to the host memory

            synthesized_fft += scatter_potential_fft3d
            synthesized_weight += scatter_potential_fft3d != 0

            if hermite:
                conj_scatter_potential_fft_3d = xp.conjugate(xp.flip(scatter_potential_fft3d, axis=(0, 1, 2)))

                del scatter_potential_fft3d

                synthesized_fft += conj_scatter_potential_fft_3d
                synthesized_weight += conj_scatter_potential_fft_3d != 0

                del conj_scatter_potential_fft_3d

        synthesized_weight -= synthesized_weight > 1
        synthesized_fft /= synthesized_weight

        factor = xp.array(
            [self.params.F2S**2 * self.params.F2Sz / (2 * xp.pi) ** (3 / 2)],
            dtype=precision.get_complex_precision(),
        )  # last factor is to adjust to the non-Unitary derivation in Tamamitsu's paper

        synthesized_array = (
            xp.fft.ifftn(xp.fft.ifftshift(synthesized_fft), norm="ortho").astype(precision.get_complex_precision())
            * factor
        )

        assert (
            synthesized_array.dtype == precision.get_complex_precision()
        ), f"{synthesized_array.dtype=}, {precision.get_complex_precision()=}"

        return synthesized_array, synthesized_fft

    def iterative_reconstruct(
        self,
        array3d: NDArray,
        array3dfft: NDArray,
        n_iter: int = 100,
        epsilon: float = 0.1,
        interval: int = 10,
        positive: bool = True,
        precision: ArrayPrecision | None = None,
    ) -> NDArray:
        """Iterative reconstruction

        Args:
            array3d (NDArray): 3D array to be reconstructed
            array3dfft (NDArray): Fourier transform of the 3D array
            n_iter (int, optional): maximum number of iterations. Defaults to 100.
            epsilon (float, optional): epsilon for the convergence. Defaults to 0.1.
            interval (int, optional): interval to print the error. Defaults to 100.
            positive (bool, optional): whether to use positive constraint. Defaults to True.
            precision (ArrayPrecision | None, optional): precision of the array. Defaults to None.

        Returns:
            NDArray: reconstructed 3D array
        """
        assert isinstance(self.params, ODTParameters)
        assert array3d.dtype == precision.get_complex_precision()
        assert array3dfft.dtype == precision.get_complex_precision()
        print("Iterative reconstruction...")
        tmp_array = array3d.copy()
        last_array = array3d.copy()
        del array3d

        factorS2F = xp.array(
            [self.params.S2F**2 * self.params.S2Fz * (2 * xp.pi) ** (3 / 2)],
            dtype=precision.get_complex_precision(),
        )
        factorF2S = xp.array(
            [self.params.F2S**2 * self.params.F2Sz / (2 * xp.pi) ** (3 / 2)],
            dtype=precision.get_complex_precision(),
        )

        err = np.inf
        count = 0
        while (err > epsilon) and (count < n_iter):
            if positive:
                tmp_array[xp.real(tmp_array) > 0] = 0
            else:
                tmp_array[xp.real(tmp_array) < 0] = 0

            tmp_fft = xp.fft.fftshift(xp.fft.fftn(tmp_array, norm="ortho")) * factorS2F
            if tmp_fft.dtype != precision.get_complex_precision():
                tmp_fft = tmp_fft.astype(precision.get_complex_precision())
            tmp_fft[array3dfft != 0] = array3dfft[array3dfft != 0]
            tmp_array = xp.fft.ifftn(xp.fft.ifftshift(tmp_fft), norm="ortho") * factorF2S
            if tmp_array.dtype != precision.get_complex_precision():
                tmp_array = tmp_array.astype(precision.get_complex_precision())

            err = calc_normalized_L2error(tmp_array, last_array)
            last_array = tmp_array

            count += 1
            if count % interval == 0:
                # err = calc_normalized_L2error(array3d, last_array)
                print(f"iter: {count}, error: {err}")

        return tmp_array, tmp_fft

    def QPI(self, pkl_format: bool = False, precision: ArrayPrecision | None = None) -> tuple[NDArray, NDArray]:
        """Quantitative phase imaging (QPI) calculation

        Args:
            pkl_format (bool, optional): whether to use the data in pickle format(compressed). Defaults to False.

        Returns:
            tuple[NDArray, NDArray]: synthesized QPI and its Fourier transform
        """
        if pkl_format:
            self.get_field_from_compressed()
        else:
            self.get_field()
        self.get_div_field_and_spectrum()
        synthesized_array, synthesized_fft = self.synthesize_spectrums(precision)
        synthesized_qpi = xp.angle(synthesized_array)

        return synthesized_qpi, synthesized_fft

    def MIPQPI(
        self, crop_center: bool = False, c_r: int = 5, pkl_format: bool = False, precision: ArrayPrecision | None = None
    ) -> tuple[NDArray, NDArray]:
        """Mid-infrared Photothermal Quantitative Phase imaging (MIPQPI) calculation

        Args:
            crop_center (bool, optional): whether to crop the center of the array for MIPQPI. Defaults to False.
            c_r (int, optional): radius of the center crop for MIPQPI. Defaults to 5.
            pkl_format (bool, optional): whether to use the data in pickle format(compressed). Defaults to False.

        Returns:
            tuple[NDArray, NDArray]: synthesized MIPQPI and its Fourier transform
        """
        if pkl_format:
            self.get_field_from_compressed()
        else:
            self.get_field()
        self.get_div_field_and_spectrum(MIP=True, crop_center=crop_center, c_r=c_r)
        synthesized_array, synthesized_fft = self.synthesize_spectrums(precision)
        synthesized_mipqpi = xp.angle(synthesized_array)

        return synthesized_mipqpi, synthesized_fft

    def ODT(
        self,
        approx: str = "Rytov",
        hermite=False,
        pkl_format: bool = False,
        iterative: bool = False,
        precision: ArrayPrecision | None = None,
        expand: bool = False,
        **kwargs,
    ) -> tuple[NDArray, NDArray]:
        """Optical Diffraction Tomography (ODT) calculation

        Args:
            approx (str, optional): approximation for the ODT calculation. Defaults to "Rytov".
            hermite (bool, optional): whether to use Hermite symmetry. Defaults to False.
            pkl_format (bool, optional): whether to use the data in pickle format(compressed). Defaults to False.
            iterative (bool, optional): whether to use iterative reconstruction. Defaults to False.
            precision (ArrayPrecision | None, optional): precision of the array. Defaults to None.
            expand (bool, optional): whether to use z-zeropadding. Defaults to False.

        Returns:
            tuple[NDArray, NDArray]: synthesized complex refractive index and its Fourier transform
        """
        assert isinstance(self.params, ODTParameters)
        if pkl_format:
            self.get_field_from_compressed()
        else:
            self.get_field()
        self.get_scattering_field(approx=approx)
        synthesized_array, synthesized_fft = self.synthesize_on_3d(hermite=hermite, precision=precision)

        if iterative:
            synthesized_array, synthesized_fft = self.iterative_reconstruct(
                synthesized_array, synthesized_fft, precision=precision, **kwargs
            )

        synthesized_array = xp.fft.fftshift(synthesized_array, axes=(2))

        print("Calculating refractive index...")
        r_index = (
            calc_refractive_index_square(synthesized_array, self.params, precision=precision) ** 0.5 - self.params.n_sol
        )
        del synthesized_array

        if expand:
            print("Expanding the z-axis...")
            r_index = zeropad_higher_kz(r_index, self.params.aperturesize - self.params.fz_extent)

        return r_index, synthesized_fft

    def MIPODT(
        self,
        approx: str = "Rytov",
        hermite=False,
        crop_center=False,
        c_r: int = 5,
        pkl_format: bool = False,
        iterative: bool = False,
        precision: ArrayPrecision | None = None,
        expand: bool = False,
        **kwargs,
    ):
        assert isinstance(self.params, ODTParameters)
        if pkl_format:
            self.get_field_from_compressed()
        else:
            self.get_field()
        self.get_scattering_field(approx=approx, MIP=True, crop_center=crop_center, c_r=c_r)

        synthesized_array, synthesized_fft = self.synthesize_on_3d(hermite=hermite, precision=precision)

        if iterative:
            synthesized_array, synthesized_fft = self.iterative_reconstruct(
                synthesized_array, synthesized_fft, precision=precision, **kwargs
            )

        synthesized_array = xp.fft.fftshift(synthesized_array, axes=(2))

        print("Calculating refractive index...")
        r_index = (
            calc_refractive_index_square(synthesized_array, self.params, precision=precision) ** 0.5 - self.params.n_sol
        )
        del synthesized_array

        if expand:
            print("Expanding the z-axis...")
            r_index = zeropad_higher_kz(r_index, self.params.aperturesize - self.params.fz_extent)

        return r_index, synthesized_fft


####################################################
# Methods for Optical Diffraction Tomography (ODT) #
####################################################


# TODO: make light-weight version of this method
def map_aperture_to_Ewald(
    array: NDArray,
    shape: tuple[int, int, int],
    oblique_shift: tuple[int, int],
    params: ODTParameters,
    precision: ArrayPrecision | None = None,
) -> NDArray:
    """map 2d array to 3d array(ODT)

    Args:
        array (NDArray): 2D array to be projected. array size should be 2*aperturesize+1 square.
        shape (tuple[int, int, int]): shape of the output 3d array. xy shape must be consistent with the input array
        oblique_shift (tuple): oblique shift in the Fourier space
        params (ODTParameters): parameters for the ODT system
        precision (ArrayPrecision, optional): precision of the array. Defaults to None.

    Returns:
        NDArray: 3D array projected to the Ewald sphere
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
    center_x = params.aperturesize - mcfg.EDGE_SIZE // 2
    center_y = params.aperturesize - mcfg.EDGE_SIZE // 2
    circle = (xx - center_x + oblique_shift[0]) ** 2 + (yy - center_y + oblique_shift[1]) ** 2
    circle = circle < (params.aperturesize // 2) ** 2
    Fz_circle = xp.sqrt(
        params.fi_mag**2 - (xx - center_x + oblique_shift[0]) ** 2 - (yy - center_y + oblique_shift[1]) ** 2
    ) - xp.sqrt(params.fi_mag**2 - oblique_shift[0] ** 2 - oblique_shift[1] ** 2)

    Fz_value = (Fz_circle + shape[2] // 2) * circle
    Fz_tile = xp.tile(Fz_value, (shape[2], 1, 1))
    Fz_tile = Fz_tile.transpose(1, 2, 0)

    Fz_tile = Fz_tile.astype(precision.get_int_precision())

    Fz_tile -= Fz_tile == 0  # to avoid 0 index match with zz

    Fz_index = zz == Fz_tile

    array_tiled = xp.stack([array] * shape[2], axis=-1)
    array_projected = array_tiled * Fz_index
    return array_projected


def calc_kz_value(
    params: ODTParameters,
    oblique_shift: tuple[int, int],
    precision: ArrayPrecision | None = None,
) -> NDArray:
    xx, yy = xp.meshgrid(
        xp.arange(2 * params.aperturesize + 1 - mcfg.EDGE_SIZE),
        xp.arange(2 * params.aperturesize + 1 - mcfg.EDGE_SIZE),
        indexing="ij",
    )
    disk = (xx - params.aperturesize + oblique_shift[0]) ** 2 + (yy - params.aperturesize + oblique_shift[1]) ** 2
    disk_mask = disk < (params.aperturesize // 2) ** 2
    fz_disk = (params.fi_mag**2 - disk) * disk_mask
    fz_disk = fz_disk**0.5
    kz_disk = fz_disk * params.k_per_pixel

    kz_disk = kz_disk.astype(precision.get_float_precision())

    return kz_disk


def calc_refractive_index_square(
    array3d: NDArray, params: ODTParameters, precision: ArrayPrecision | None = None
) -> NDArray:
    """calculate :math:`n^2` for the ODT calculation

    Args:
        array3d (NDArray): Array to calculate refractive index square
        params (ODTParameters): parameters for the ODT system

    Returns:
        NDArray: complex refractive index
    """
    r_3d_square = params.n_sol**2 * (
        xp.ones(array3d.shape, dtype=precision.get_complex_precision())
        - array3d / (params.fi_mag * params.k_per_pixel) ** 2
    )
    return r_3d_square


def discard_z(array: NDArray, threshold: int) -> NDArray:
    """internal method for `discard_higher_kz`. discard the higher z values

    Args:
        array (NDArray): original array(usually Fourier transformed array)
        threshold (int): cut off threshold

    Returns:
        NDArray: lowpassed array
    """
    array = array[:, :, threshold : array.shape[2] - threshold]
    return array


def discard_higher_kz(array: NDArray, threshold: int) -> NDArray:
    """discard the higher z values

    Args:
        array (NDArray): original array
        threshold (int): cut off threshold

    Returns:
        NDArray: lowpassed array
    """
    norm_factor = xp.sqrt((array.shape[2] - 2 * threshold) / array.shape[2])
    array_fft = xp.fft.fftshift(xp.fft.fftn(array, norm="ortho"))
    discarded = discard_z(array_fft, threshold)
    new_array = xp.fft.ifftn(xp.fft.ifftshift(discarded), norm="ortho") * norm_factor
    return new_array


def zeropad_higher_kz(array: NDArray, extend: int, precision: ArrayPrecision, gpu_on: bool = True) -> NDArray:
    """zero pad the higher z values

    Args:
        array (NDArray): original array
        extend (int): extend size
        precision (ArrayPrecision): precision of the array
        gpu_on (bool, optional): whether to use GPU if available. Defaults to True.

    Returns:
        NDArray: zero padded array
    """
    if gpu_on & mcfg._cp:
        executor = cp
    elif (not gpu_on) & mcfg._cp:
        array = cp.asnumpy(array)
        executor = np
    else:
        executor = xp
    norm_factor = executor.array(
        [executor.sqrt((array.shape[2] + 2 * extend) / array.shape[2])], dtype=precision.get_float_precision()
    )
    array_fft = executor.fft.fftshift(executor.fft.fftn(array, norm="ortho")).astype(precision.get_complex_precision())
    del array
    array_fft = executor.pad(array_fft, ((0, 0), (0, 0), (extend, extend)), mode="constant", constant_values=0)
    assert array_fft.dtype == precision.get_complex_precision()
    array = (
        executor.fft.ifftn(executor.fft.ifftshift(array_fft), norm="ortho").astype(precision.get_complex_precision())
        * norm_factor
    )
    return array


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


def calc_normalized_L2error(array: NDArray, ref_array: NDArray) -> float:
    """Calculate the normalized L2 error

    Args:
        array (NDArray): array to be compared
        ref_array (NDArray): reference array

    Returns:
        float: normalized L2 error
    """
    error = xp.sum(xp.abs(array - ref_array)) / xp.sum(xp.abs(ref_array))
    return error
