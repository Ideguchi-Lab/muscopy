"""Quantitative Phase Imaging (QPI) and MIP-QPI."""

from __future__ import annotations

import cmath
import inspect
from dataclasses import dataclass, fields
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import types
    from collections.abc import Iterable

    from muscopy.backend_manager import _T, ArrayProtocol
    from muscopy.cfg import OffsetRegions, Region


@dataclass
class QPIParameters:
    r"""QPI parameters.

    Parameters
    ----------
    na : `float`
        Numerical aperture of the objective lens
    wavelength_m : `float`
        Wavelength of the light in meters
    img_size_px : `int`
        Size of the image in pixels. We assume square image.
    px_size_m : `float`
        Pixel size in meters
    n_sol : `float`
        Refractive index of the solution
    """

    na: float
    wavelength_m: float
    img_size_px: int
    px_size_m: float
    n_sol: float

    @cached_property
    def img_center(self) -> tuple[int, int]:
        r"""Get the center position of the image.

        Returns
        -------
        `tuple`\[`int`, `int`\]
            The center position of the image
        """
        return (self.img_size_px // 2, self.img_size_px // 2)

    @cached_property
    def freq_per_px(self) -> float:
        """Frequency(1/meter) per pixel in the Fourier space.

        Returns
        -------
        `float`
            Frequency(1/meter) per pixel in the Fourier space
        """
        return 1 / (self.px_size_m * self.img_size_px)

    @cached_property
    def k_per_px(self) -> float:
        r"""Get the wave vector per pixel in the Fourier space.

        Returns
        -------
        `float`
            the wave vector per pixel in the Fourier space
        """
        return 2 * np.pi * self.freq_per_px

    @cached_property
    def aperturesize_px(self) -> int:
        """Get the size of the aperture in pixel unit.

        Returns
        -------
        `int`
            The size of the aperture in pixel unit
        """
        return 2 * round(self.na / self.wavelength_m / self.freq_per_px) + 1

    @cached_property
    def light_freq_px(self) -> float:
        r"""Get the light frequency in pixel unit.

        Returns
        -------
        `float`
            the magnitude of the light frequency in the pixel unit
        """
        return self.n_sol / self.wavelength_m / self.freq_per_px

    @cached_property
    def imgpx_m_per_px(self) -> float:
        """Get the size of the imaging pixel(QPI pixel) in meter unit.

        Returns
        -------
        `float`
            The size of the imaging pixel(QPI pixel) in meter unit
        """
        return self.px_size_m * self.img_size_px / self.aperturesize_px

    @cached_property
    def hologram2fourier(self) -> float:
        """Fourier factor from hologram to spectrum.

        Returns
        -------
        `float`
            factor from hologram to spectrum
        """
        return (self.px_size_m / self.freq_per_px) ** 0.5

    @cached_property
    def fourier2cpfield(self) -> float:
        """Fourier factor from spectrum to complex field.

        Returns
        -------
        `float`
            factor from spectrum to complex field
        """
        return (self.freq_per_px / self.imgpx_m_per_px) ** 0.5

    @cached_property
    def cpfield2spectrum(self) -> float:
        """Fourier factor from complex field to spectrum.

        Returns
        -------
        `float`
            factor from complex field to spectrum
        """
        return (self.imgpx_m_per_px / self.freq_per_px) ** 0.5


def print_qpi_all_parameters(param: QPIParameters, *, show_properties: bool = False) -> None:
    """Print all parameters of QPIParameters dataclass.

    Parameters
    ----------
    param : `QPIParameters`
        QPIParameters dataclass instance
    show_properties : `bool`, optional
        Show properties or not, by default `False`
    """
    print("=== Dataclass Parameters ===")  # noqa: T201
    for field_obj in fields(param):
        name = field_obj.name
        value = getattr(param, name)
        print(f"{name}: {value}")  # noqa: T201

    if show_properties:
        # print property and cached_property
        print("\n=== Properties ===")  # noqa: T201
        # detect properties and cached_properties
        prop_members = dict(inspect.getmembers(type(param), lambda m: isinstance(m, (property, cached_property))))

        dataclass_field_names = {field_obj.name for field_obj in fields(param)}
        for name in prop_members:
            # check if the name is not in dataclass fields
            if name not in dataclass_field_names:
                print(f"{name}: {getattr(param, name)}")  # noqa: T201


def make_disk(
    backend: types.ModuleType,
    center: tuple[int, int],
    radius: float,
    array_shape: int | tuple[int, int],
    highpass: bool = False,
) -> ArrayProtocol[bool]:
    r"""Make a disk mask with specified center and radius.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    center : `tuple`\[`int`, `int`\]
        The center position of the disk mask
    radius : `float`
        The radius of the disk mask
    array_shape : `int` | `tuple`\[`int`, `int`\]
        The shape of the array. If int, it is assumed to be square.
    highpass : `bool`, optional
        Apply highpass mask or not, by default False

    Returns
    -------
    `ArrayProtocol`\[`bool`\]
        The disk mask with specified center and radius
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = backend.meshgrid(backend.arange(array_shape[0]), backend.arange(array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return circle > radius**2 if highpass else circle < radius**2


def _crop_array(array: ArrayProtocol[_T], center: tuple[int, int], width: int) -> ArrayProtocol[_T]:
    return array[
        center[0] - width // 2 : center[0] + width // 2 + 1,
        center[1] - width // 2 : center[1] + width // 2 + 1,
    ]


def get_spectrum(  # noqa: PLR0913
    backend: types.ModuleType,
    ft_array: ArrayProtocol,
    params: QPIParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> ArrayProtocol:
    r"""Get the spectrum of the hologram array.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    ft_array : `ArrayProtocol`
        Fourier spectrum of the hologram array
    params : `QPIParameters`
        QPIParameters class
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of off-axis digital holography
    crop_center : `bool`, optional
        Whether to crop center or not, by default False
        This option is used for MIP-QPI
    c_r : `int`, optional
        The crop radius, by default 5

    Returns
    -------
    `ArrayProtocol`
        The spectrum of complex amplitude
    """
    mask = make_disk(backend, offaxis_center, params.aperturesize_px // 2, params.img_size_px)
    ft_array *= mask

    if crop_center:
        mask_highpass = make_disk(backend, offaxis_center, c_r, params.img_size_px, highpass=True)
        ft_array *= mask_highpass

    return _crop_array(ft_array, offaxis_center, params.aperturesize_px)


def get_spectrums(  # noqa: PLR0913
    backend: types.ModuleType,
    array: ArrayProtocol,
    params: QPIParameters,
    offaxis_centers: Iterable[tuple[int, int]],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> list[ArrayProtocol]:
    r"""Get the spectrums of the complex fileds.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    array : `ArrayProtocol`
        Hologram array
    params : `QPIParameters`
        QPIParameters class
    offaxis_centers : `Iterable`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography
    crop_center : `bool`, optional
        Whether to crop center or not, by default False
        This option is used for MIP-QPI
    c_r : `int`, optional
        The crop radius, by default 5

    Returns
    -------
    `list`\[`ArrayProtocol`\]
        The spectrums of complex amplitude
    """
    hologram_fft = backend.fft.fftshift(backend.fft.fft2(array)) * params.hologram2fourier
    cp_spectrums = []
    for offaxis_center in offaxis_centers:
        cp_spectrum = get_spectrum(backend, hologram_fft, params, offaxis_center, crop_center=crop_center, c_r=c_r)
        cp_spectrums.append(cp_spectrum)
    return cp_spectrums


def correct_offset(
    backend: types.ModuleType,
    array: ArrayProtocol,
    offset_regs: OffsetRegions,
    *,
    phase: bool = True,
    amplitude: bool = True,
) -> ArrayProtocol:
    """Correct the phase and amplitude offset of the array.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    array : `ArrayProtocol`
        Complex amplitude array
    offset_regs : `OffsetRegions`
        The regions to calculate the offset
    phase : `bool`, optional
        Correct Phase offset, by default True
    amplitude : `bool`, optional
        Correct Amplitude scaling, by default True

    Returns
    -------
    `ArrayProtocol`
        The corrected array
    """
    if offset_regs is None:
        return array

    phase_offset_list = []
    amplitude_offset_list = []
    for region in offset_regs:
        phase_offset_list.append(
            backend.mean(backend.angle(array[region[0][0] : region[0][1], region[1][0] : region[1][1]]))
        )
        amplitude_offset_list.append(
            backend.mean(backend.abs(array[region[0][0] : region[0][1], region[1][0] : region[1][1]]))
        )

    phase_offset = backend.mean(backend.array(phase_offset_list)) if phase else 0
    amplitude_scale = backend.mean(backend.array(amplitude_offset_list)) if amplitude else 1

    return array * cmath.exp(-1j * phase_offset) / amplitude_scale


def qpi(
    backend: types.ModuleType,
    array: ArrayProtocol,
    reference: ArrayProtocol,
    params: QPIParameters,
    offaxis_centers: Iterable[tuple[int, int]],
) -> list[ArrayProtocol]:
    r"""Calculate the QPI phase image.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    array : `ArrayProtocol`
        Hologram array
    reference : `ArrayProtocol`
        Reference hologram array
    params : `QPIParameters`
        QPIParameters class
    offaxis_centers : `Iterable`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography

    Returns
    -------
    `list`\[`ArrayProtocol`\]
        The QPI phase image

    Raises
    ------
    ValueError
        If the array and reference have different shapes
    """
    if array.shape != reference.shape:
        msg = "Array and reference must have the same shape"
        raise ValueError(msg)

    ft_array = backend.fft.fftshift(backend.fft.fft2(array)) * params.hologram2fourier
    ft_reference = backend.fft.fftshift(backend.fft.fft2(reference)) * params.hologram2fourier
    spectrums = get_spectrums(backend, ft_array, params, offaxis_centers)
    ref_spectrums = get_spectrums(backend, ft_reference, params, offaxis_centers)

    cp_fields = []

    for spectrum, ref_spectrum in zip(spectrums, ref_spectrums):
        cp_field = backend.fft.ifft2(backend.fft.ifftshift(spectrum)) * params.fourier2cpfield
        ref_cp_field = backend.fft.ifft2(backend.fft.ifftshift(ref_spectrum)) * params.fourier2cpfield
        cp_field /= ref_cp_field
        cp_fields.append(cp_field)

    return [backend.angle(cp_field) for cp_field in cp_fields]


def mip_qpi(  # noqa: PLR0913
    backend: types.ModuleType,
    array_on: ArrayProtocol,
    array_off: ArrayProtocol,
    params: QPIParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
    mip_center_reg: Region | None = None,
) -> ArrayProtocol:
    r"""Calculate the MIP-QPI phase image.

    Parameters
    ----------
    array_on : `ArrayProtocol`
        MIR ON hologram array
    array_off : `ArrayProtocol`
        MIR OFF hologram array
    params : `QPIParameters`
        QPIParameters class
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of off-axis digital holography
    crop_center : `bool`, optional
        Whether to crop center or not, by default False
        This option is used for MIP-QPI
    c_r : `int`, optional
        The crop radius, by default 5
    mip_center_reg : `Region` | `None`, optional
        _description_, by default None

    Returns
    -------
    `ArrayProtocol`
        The MIP-QPI phase image

    Raises
    ------
    ValueError
        If the array on and off have different shapes
    """
    if array_on.shape != array_off.shape:
        msg = "Array on and off must have the same shape"
        raise ValueError(msg)

    ft_array_on = backend.fft.fftshift(backend.fft.fft2(array_on)) * params.hologram2fourier
    ft_array_off = backend.fft.fftshift(backend.fft.fft2(array_off)) * params.hologram2fourier

    spectrum_on = get_spectrum(backend, ft_array_on, params, offaxis_center, crop_center=crop_center, c_r=c_r)
    spectrum_off = get_spectrum(backend, ft_array_off, params, offaxis_center, crop_center=crop_center, c_r=c_r)

    cp_field_on = backend.fft.ifft2(backend.fft.ifftshift(spectrum_on)) * params.fourier2cpfield
    cp_field_off = backend.fft.ifft2(backend.fft.ifftshift(spectrum_off)) * params.fourier2cpfield

    array_div = cp_field_on / cp_field_off

    if mip_center_reg is not None:
        center_phase = backend.mean(
            backend.angle(
                array_div[
                    mip_center_reg[0][0] : mip_center_reg[0][1],
                    mip_center_reg[1][0] : mip_center_reg[1][1],
                ]
            )
        )
        if center_phase < 0:
            array_div = 1 / array_div

    return backend.angle(array_div)


def _get_dc_ac(
    hologram: NDArray,
    params: QPIParameters,
) -> tuple[NDArray, NDArray]:
    """Calculates the DC intensity of the object.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.

    Returns
    -------
        NDArray: The DC and AC component of the object.
    """
    scale_factor = params.aperturesize / params.img_shape[0]
    fft = xp.fft.fftshift(xp.fft.fft2(hologram))
    dc_disk = make_disk(params.img_center, params.aperturesize // 2, params.img_shape)
    ac_disk = make_disk(params.offaxis_center, params.aperturesize // 2, params.img_shape)

    dc_fft = fft * dc_disk
    ac_fft = fft * ac_disk

    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize)
    ac_cropped = crop_array(ac_fft, params.offaxis_center, params.aperturesize)

    dc = xp.fft.ifft2(xp.fft.ifftshift(dc_cropped)) * scale_factor**2
    ac = xp.fft.ifft2(xp.fft.ifftshift(ac_cropped)) * scale_factor**2

    return dc, ac


def _get_visibility(
    dc: NDArray,
    ac: NDArray,
) -> NDArray:
    """Calculates the visibility of the object.

    Args:
        dc (NDArray): dc component of the hologram
        ac (NDArray): ac component of the hologram

    Returns
    -------
        NDArray: pixel-wise visibility of the hologram
    """
    visibility = xp.abs(ac) / xp.abs(dc) * 2
    return visibility


def _get_phase_noise(
    visibility: NDArray,
    aperturesize: int,
    dc_intensity: NDArray,
    sensorsize: tuple[int, int],
    sensor_noise: int = 0,
) -> NDArray:
    """Calculates the phase noise in a given array.

    Here, dc_intensity is the photon number of DC components, as the shot noise is proportional to the square root of the photon number.

    Args:
        visibility (NDArray): The visibility of the object.
        aperturesize (int): The size of the aperture.
        dc_intensity (NDArray): The DC intensity of the object.
        sensorsize (tuple[int, int]): The size of the sensor.
        sensor_noise (int, optional): The sensor noise (unit: e-). Defaults to 0.

    Returns
    -------
        NDArray: The phase noise
    """
    if not visibility.shape == dc_intensity.shape:
        raise ValueError("Visibility and DC intensity must have the same shape")
    aperture_area = xp.pi * (aperturesize / 2) ** 2
    sensor_area = sensorsize[0] * sensorsize[1]
    phase_noise = xp.sqrt(
        2 * aperture_area * (dc_intensity + sensor_noise**2) / (visibility**2 * dc_intensity**2 * sensor_area)
    )

    return phase_noise


def calc_visibility(
    hologram: NDArray,
    params: QPIParameters,
) -> NDArray:
    """Calculates the visibility of the object.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.

    Returns
    -------
        NDArray: pixel-wise visibility of the hologram
    """
    dc, ac = _get_dc_ac(hologram, params)
    visibility = _get_visibility(dc, ac)

    return visibility


def calc_phase_noise(
    hologram: NDArray,
    params: QPIParameters,
    fullwell: float,
    bit_depth: int,
    sensor_noise: int = 0,
) -> NDArray:
    """Calculates the phase noise in a given array.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.
        fullwell (float): The full well capacity of the sensor.
        bit_depth (int): The bit depth of the sensor.
        sensor_noise (int): The sensor noise (unit: e-). Defaults to 0.

    Returns
    -------
        NDArray: The phase noise
    """
    dc, ac = _get_dc_ac(hologram, params)
    visibility = _get_visibility(dc, ac)
    dc_factor = fullwell / (2**bit_depth)
    phase_noise = _get_phase_noise(
        visibility,
        params.aperturesize,
        dc_factor * xp.abs(dc),
        params.img_shape,
        sensor_noise,
    )

    return phase_noise
