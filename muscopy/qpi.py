from __future__ import annotations

from typing import NewType

import numpy as np

try:
    import cupy as xp

    _cp = True
except ImportError:
    import numpy as xp

    _cp = False

import muscopy.cfg as mcfg
from muscopy.cfg import OffsetRegions


class QPIParameters:
    def __init__(
        self,
        wavelength: float,
        NA: float,
        img_shape: tuple[int, int],
        pixelsize: float,
        offaxis_center: tuple[int, int],
        n_sol: float = 1.33,
    ):
        """QPIParameters class for QPI calculation

        Args:
            wavelength (float): wavelength of the probe
            NA (float): Numerical aperture of the objective
            img_shape (tuple[int, int]): shape of the image
            pixelsize (float): image pixel size
            offaxis_center (tuple[int, int]): center position of the off axis holography in the Fourier domain
            n_sol (float, optional): refractive index of the solvent. Defaults to 1.33.
        """
        self.wav = wavelength
        self.NA = NA
        self.img_shape = img_shape
        self.pixelsize = pixelsize
        self.offaxis_center = offaxis_center
        self.n_sol = n_sol

        self._calc_params()

    def _calc_params(self):
        """Calculate parameters for QPI calculation. This method is called in __init__ method."""

        self.img_center = (self.img_shape[0] // 2, self.img_shape[1] // 2)
        self.dim = self.img_shape[0]
        self.freq_per_pixel = 1 / (self.pixelsize * self.dim)  # 1 / L
        self.k_per_pixel = 2 * np.pi * self.freq_per_pixel  # unit of k in terms of pixel unit
        self.aperturesize = 2 * round(self.NA / self.wav / self.freq_per_pixel) + 1  # 2 * f_BW + 1
        self.fi_mag = self.n_sol / self.wav / self.freq_per_pixel  # |k| in terms of pixel unit

        self.imgpx_unit = (
            self.pixelsize * self.img_shape[0] / (2 * self.aperturesize + 1 - mcfg.EDGE_SIZE)
        )  # unit image pixel size on the cropped image plane

    def print_all_parameters(self):
        """Print all parameters in the QPIParameters class"""
        for key, value in vars(self).items():
            print(f"{key}={value}")

    def Hologram2F(self) -> float:
        """Factor to convert the hologram to the Fourier space

        Returns:
            float: FFT factor
        """
        return (self.pixelsize / self.k_per_pixel) ** 0.5

    def S2F(self) -> float:
        """Factor to convert the spatial domain to the Fourier space

        Returns:
            float: FFT factor
        """
        return (self.imgpx_unit / self.k_per_pixel) ** 0.5

    def F2S(self) -> float:
        """Factor to convert the Fourier space to the spatial domain

        Returns:
            float: FFT factor
        """
        return (self.k_per_pixel / self.imgpx_unit) ** 0.5


def make_disk(center: tuple[int, int], radius: float, array_shape: tuple[int, int], highpass: bool = False) -> xp.array:
    """make disk mask for filtering

    Args:
        center (tuple[int, int]): center position of the disk mask
        radius (float): radius of the disk mask
        array_shape (tuple[int, int]): shape of the array
        highpass (bool, optional): Filter low frequency or not. Defaults to False.

    Returns:
        xp.array: disk mask
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = xp.meshgrid(xp.arange(array_shape[0]), xp.arange(array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def crop_array(array: xp.array, center: tuple[int, int], width: int) -> xp.array:
    """internal method. crop the array with specified center and width.

    Args:
        array (xp.array): array to be cropped
        center (tuple of int): center position of the cropped array
        width (int): width of the cropped array

    Returns:
        xp.array: cropped array
    """
    return array[
        center[0] - width // 2 : center[0] + width // 2 + 1,
        center[1] - width // 2 : center[1] + width // 2 + 1,
    ]


def get_spectrum(array: xp.array, params: QPIParameters, crop_center: bool = False, c_r: int = 5) -> xp.array:
    """internal method. get the spectrum of the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: cropped spectrum of the hologram array
    """
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    mask = make_disk(params.offaxis_center, params.aperturesize / 2, params.img_shape)
    array_fft = array_fft * mask

    if crop_center:
        mask_highpass = make_disk(params.offaxis_center, c_r, params.img_shape, highpass=True)
        array_fft = array_fft * mask_highpass

    array_fft = crop_array(array_fft, params.offaxis_center, params.aperturesize)

    return array_fft


def get_field(array: xp.array, params: QPIParameters, crop_center: bool = False, c_r: int = 5) -> xp.array:
    """internal method. get the electric field from the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: field from the hologram array
    """
    array_fft = get_spectrum(array, params, crop_center, c_r)
    array = xp.fft.ifft2(xp.fft.ifftshift(array_fft))
    return array


def correct_offset(array, offset_regs: OffsetRegions, phase: bool = True, amplitude: bool = True) -> xp.array:
    """internal method. correct phase and amplitude offset

    Args:
        array (NDArray): input complex array
        offset_regs (OffsetRegions): regions for offset calculation

    Returns:
        xp.array: corrected array
    """
    if offset_regs is None:
        return array
    phase_offset_list = []
    amplitude_offset_list = []
    for region in offset_regs:
        phase_offset_list.append(xp.mean(xp.angle(array[region[0][0] : region[0][1], region[1][0] : region[1][1]])))
        amplitude_offset_list.append(xp.mean(xp.abs(array[region[0][0] : region[0][1], region[1][0] : region[1][1]])))
    if phase:
        phase_offset = xp.mean(xp.array(phase_offset_list))  #  - xp.pi / 2
    else:
        phase_offset = 0
    if amplitude:
        amplitude_offset = xp.mean(xp.array(amplitude_offset_list))
    else:
        amplitude_offset = 1

    array = array * xp.exp(-1j * phase_offset) / amplitude_offset

    return array


def QPI(
    array: xp.array,
    reference: xp.array,
    params: QPIParameters,
) -> xp.array:
    """Quantitative phase imaging (QPI) calculation

    Args:
        array (xp.array): on-axis hologram
        reference (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class

    Returns:
        xp.array: QPI phase image
    """
    assert array.shape == reference.shape

    array_field = get_field(array, params)
    ref_array_field = get_field(reference, params)

    array_div = array_field / ref_array_field

    # remove phase and amplitude offset
    if mcfg.OFFSET_REGS is not None:
        array_div = correct_offset(array_div, mcfg.OFFSET_REGS)

    dif_phase = xp.angle(array_div)

    return dif_phase


def MIPQPI(
    array_on: xp.array,
    array_off: xp.array,
    params: QPIParameters,
    crop_center: bool = False,
    **kwargs,
) -> xp.array:
    """Mid-infrared photothermal quantitative phase imaging (MIP-QPI) calculation

    Args:
        array_on (xp.array): on-axis hologram
        array_off (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: MIP-QPI phase image
    """
    assert array_on.shape == array_off.shape
    array_on_field = get_field(array_on, params, crop_center=crop_center, **kwargs)
    array_off_field = get_field(array_off, params, crop_center=crop_center, **kwargs)

    array_div = array_on_field / array_off_field

    # remove phase and amplitude offset
    if mcfg.OFFSET_REGS is not None:
        array_div = correct_offset(array_div, mcfg.OFFSET_REGS, phase=True)

    dif_phase = xp.angle(array_div)

    return dif_phase
