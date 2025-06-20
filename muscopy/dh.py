"""Digital Holography module.

This module provides:

- `MuParameters`: A dataclass to hold Micorscopy parameters.
- `print_all_parameters`: A function to print all parameters of `MuParameters` dataclass.
- `make_disk`: A function to create a disk mask.
- `crop_array`: A function to crop an array.
- `get_spectrum`: A function to get the spectrum of the hologram array.
- `get_spectrums`: A function to get the spectrums of the complex fields.
- `correct_offset`: A function to correct the phase and amplitude offset of the array.
- `offaxis_dh`: A function to reconstruct the complex wave front using off-axis digital holography.
- `demultiplex_cp_arrays`: A function to demultiplex a set of CP arrays using a demultiplexing matrix.
"""

from __future__ import annotations

import cmath
import dataclasses
import inspect
import math
import typing
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from muscopy.cfg import OffsetRegions


@dataclasses.dataclass
class MuParameters:
    r"""Microscopy parameters.

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

    def verify_parameters(self) -> None:
        """Verify the parameters.

        Raises
        ------
        ValueError
            1. If the NA is negative.
            2. If the wavelength is negative.
            3. If the image size is negative.
            4. If the pixel size is negative.
            5. If the refractive index of the solution is negative.
            6. If the NA is greater than the refractive index of the solution.
        """
        if self.na < 0:
            msg = "NA cannot be negative."
            raise ValueError(msg)
        if self.wavelength_m < 0:
            msg = "Wavelength cannot be negative."
            raise ValueError(msg)
        if self.img_size_px < 0:
            msg = "Image size cannot be negative."
            raise ValueError(msg)
        if self.px_size_m < 0:
            msg = "Pixel size cannot be negative."
            raise ValueError(msg)
        if self.n_sol < 0:
            msg = "Refractive index of the solution cannot be negative."
            raise ValueError(msg)
        if self.na > self.n_sol:
            msg = "NA cannot be greater than the refractive index of the solution."
            raise ValueError(msg)

    @property
    def img_center(self) -> tuple[int, int]:
        r"""Get the center position of the image.

        Returns
        -------
        `tuple`\[`int`, `int`\]
            The center position of the image
        """
        return (self.img_size_px // 2, self.img_size_px // 2)

    @property
    def freq_per_px(self) -> float:
        """Frequency(1/meter) per pixel in the Fourier space.

        Returns
        -------
        `float`
            Frequency(1/meter) per pixel in the Fourier space
        """
        return 1 / (self.px_size_m * self.img_size_px)

    @property
    def k_per_px(self) -> float:
        r"""Get the wave vector per pixel in the Fourier space.

        Returns
        -------
        `float`
            the wave vector per pixel in the Fourier space
        """
        return 2 * math.pi * self.freq_per_px

    @property
    def aperturesize_px(self) -> int:
        """Get the size of the aperture in pixel unit.

        Returns
        -------
        `int`
            The size of the aperture in pixel unit
        """
        return 2 * round(self.na / self.wavelength_m / self.freq_per_px) + 1

    @property
    def light_freq_px(self) -> float:
        r"""Get the light frequency in pixel unit.

        Returns
        -------
        `float`
            the magnitude of the light frequency in the pixel unit
        """
        return self.n_sol / self.wavelength_m / self.freq_per_px

    @property
    def imgpx_m_per_px(self) -> float:
        """Get the size of the imaging pixel(QPI pixel) in meter unit.

        Returns
        -------
        `float`
            The size of the imaging pixel(QPI pixel) in meter unit
        """
        return self.px_size_m * self.img_size_px / self.aperturesize_px

    @property
    def hologram2spectrum(self) -> float:
        """Fourier factor from hologram to spectrum.

        Returns
        -------
        `float`
            factor from hologram to spectrum
        """
        return float((self.px_size_m / self.freq_per_px) ** 0.5)

    @property
    def spectrum2cpfield(self) -> float:
        """Fourier factor from spectrum to complex field.

        Returns
        -------
        `float`
            factor from spectrum to complex field
        """
        return float((self.freq_per_px / self.imgpx_m_per_px) ** 0.5)

    @property
    def cpfield2spectrum(self) -> float:
        """Fourier factor from complex field to spectrum.

        Returns
        -------
        `float`
            factor from complex field to spectrum
        """
        return float((self.imgpx_m_per_px / self.freq_per_px) ** 0.5)


def print_all_parameters(param: MuParameters, *, show_properties: bool = False) -> None:
    """Print all parameters of MuParameters dataclass.

    Parameters
    ----------
    param : `MuParameters`
        QPIParameters dataclass instance
    show_properties : `bool`, optional
        Show properties or not, by default `False`
    """
    print("=== Dataclass Parameters ===")  # noqa: T201
    for field_obj in dataclasses.fields(param):
        name = field_obj.name
        value = getattr(param, name)
        print(f"{name}: {value}")  # noqa: T201

    if show_properties:
        # print property and cached_property
        print("\n=== Properties ===")  # noqa: T201
        # detect properties and cached_properties
        prop_members = dict(
            inspect.getmembers(type(param), lambda m: isinstance(m, (property))),
        )

        dataclass_field_names = {field_obj.name for field_obj in dataclasses.fields(param)}
        for name in prop_members:
            # check if the name is not in dataclass fields
            if name not in dataclass_field_names:
                print(f"{name}: {getattr(param, name)}")  # noqa: T201


def make_disk(
    center: tuple[int, int],
    radius: float,
    array_shape: int | tuple[int, int],
    *,
    highpass: bool = False,
) -> Array:
    r"""Make a disk mask with specified center and radius.

    Parameters
    ----------
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
    `jax.Array`
        The disk mask with specified center and radius
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = jnp.meshgrid(jnp.arange(array_shape[0]), jnp.arange(array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return circle > radius**2 if highpass else circle <= radius**2


def crop_array(array: Array, center: tuple[int, int], width: int) -> Array:
    r"""Crop the array to the specified width around the center.

    Parameters
    ----------
    array : `jax.Array`
        The array to be cropped
    center : `tuple`\[`int`, `int`\]
        The center position of the crop
    width : `int`
        The width of the crop

    Returns
    -------
    `jax.Array`
        The cropped array
    """
    return array[
        center[0] - width // 2 : center[0] + width // 2 + 1,
        center[1] - width // 2 : center[1] + width // 2 + 1,
    ]


def get_spectrum(
    ft_array: Array,
    params: MuParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> Array:
    r"""Get the spectrum of the hologram array.

    Parameters
    ----------
    ft_array : `jax.Array`
        Fourier spectrum of the hologram array
    params : `MuParameters`
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
    `jax.Array`
        The spectrum of complex amplitude
    """
    mask = make_disk(offaxis_center, params.aperturesize_px // 2, params.img_size_px)
    ft_array *= mask

    if crop_center:
        mask_highpass = make_disk(offaxis_center, c_r, params.img_size_px, highpass=True)
        ft_array *= mask_highpass

    return crop_array(ft_array, offaxis_center, params.aperturesize_px)


def get_spectrums(
    ft_array: Array,
    params: MuParameters,
    offaxis_centers: Iterable[tuple[int, int]],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> list[Array]:
    r"""Get the spectrums of the complex fileds.

    Parameters
    ----------
    ft_array : `jax.Array`
        Fourier transformed hologram array
    params : `MuParameters`
        Microscopy Parameters class
    offaxis_centers : `Iterable`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography
    crop_center : `bool`, optional
        Whether to crop center or not, by default False
        This option is used for MIP-QPI
    c_r : `int`, optional
        The crop radius, by default 5

    Returns
    -------
    `list`\[`jax.Array`\]
        The spectrums of complex amplitude
    """
    cp_spectrums = []
    for offaxis_center in offaxis_centers:
        cp_spectrum = get_spectrum(ft_array, params, offaxis_center, crop_center=crop_center, c_r=c_r)
        cp_spectrums.append(cp_spectrum)
    return cp_spectrums


def correct_offset(
    array: Array,
    offset_regs: OffsetRegions,
    *,
    phase: bool = True,
    amplitude: bool = True,
) -> Array:
    """Correct the phase and amplitude offset of the array.

    Parameters
    ----------
    array : `jax.Array`
        Complex amplitude array
    offset_regs : `OffsetRegions`
        The regions to calculate the offset
    phase : `bool`, optional
        Correct Phase offset, by default True
    amplitude : `bool`, optional
        Correct Amplitude scaling, by default True

    Returns
    -------
    `jax.Array`
        The corrected array
    """
    if offset_regs is None:
        return array

    phase_offset_list = []
    amplitude_offset_list = []
    for region in offset_regs:
        phase_offset_list.append(
            jnp.mean(jnp.angle(array[region[0][0] : region[0][1], region[1][0] : region[1][1]])),
        )
        amplitude_offset_list.append(
            jnp.mean(jnp.abs(array[region[0][0] : region[0][1], region[1][0] : region[1][1]])),
        )

    phase_offset = jnp.mean(jnp.array(phase_offset_list)) if phase else 0
    amplitude_scale = jnp.mean(jnp.array(amplitude_offset_list)) if amplitude else 1

    return array * cmath.exp(-1j * phase_offset) / amplitude_scale


@typing.overload
def offaxis_dh(
    array: Array,
    reference: Array,
    params: MuParameters,
    offaxis_centers: tuple[int, int],
) -> Array: ...


@typing.overload
def offaxis_dh(
    array: Array,
    reference: Array,
    params: MuParameters,
    offaxis_centers: Sequence[tuple[int, int]],
) -> list[Array]: ...


def offaxis_dh(
    array: Array,
    reference: Array,
    params: MuParameters,
    offaxis_centers: tuple[int, int] | Sequence[tuple[int, int]],
) -> Array | list[Array]:
    r"""Reconstruct the complex wave front using off-axis digital holography.

    Parameters
    ----------
    array : `jax.Array`
        Hologram array
    reference : `jax.Array`
        Reference hologram array
    params : `MuParameters`
        Microscopy Parameters class
    offaxis_centers : `tuple`\[`int`, `int`\] | `collections.abc.Sequence`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography

    Returns
    -------
    `list`\[`jax.Array`\]
        The complex wave front

    Raises
    ------
    ValueError
        If the array and reference have different shapes
    """
    params.verify_parameters()
    if array.shape != reference.shape:
        msg = "Array and reference must have the same shape"
        raise ValueError(msg)

    ft_array = jnp.fft.fftshift(jnp.fft.fft2(array)) * params.hologram2spectrum
    ft_reference = jnp.fft.fftshift(jnp.fft.fft2(reference)) * params.hologram2spectrum
    if isinstance(offaxis_centers[0], int):
        offaxis_centers = typing.cast("tuple[int, int]", offaxis_centers)
        spectrum = get_spectrum(ft_array, params, offaxis_centers)
        ref_spectrum = get_spectrum(ft_reference, params, offaxis_centers)
        cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(spectrum)) * params.spectrum2cpfield
        ref_cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(ref_spectrum)) * params.spectrum2cpfield
        cp_field /= ref_cp_field
        return cp_field

    offaxis_centers = typing.cast("list[tuple[int, int]]", offaxis_centers)
    spectrums = get_spectrums(ft_array, params, offaxis_centers)
    ref_spectrums = get_spectrums(ft_reference, params, offaxis_centers)

    cp_fields = []

    for spectrum, ref_spectrum in zip(spectrums, ref_spectrums, strict=False):
        cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(spectrum)) * params.spectrum2cpfield
        ref_cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(ref_spectrum)) * params.spectrum2cpfield
        cp_field /= ref_cp_field
        cp_fields.append(cp_field)

    return cp_fields


def demultiplex_cp_arrays(
    cp_arrays: Sequence[Array],
    demultiplexing_matrix: Array,
) -> list[Array]:
    r"""Demultiplex a set of CP arrays using a demultiplexing matrix.

    Parameters
    ----------
    cp_arrays : `collections.abc.Sequence`\[`jax.Array`\]
        List of CP arrays to be demultiplexed.
    demultiplexing_matrix : `jax.Array`
        Demultiplexing coefficient matrix.

    Returns
    -------
    `list`\[`jax.Array`\]
        List of demultiplexed CP arrays.

    Raises
    ------
    ValueError
        1. If the demultiplexing matrix is not square
        2. If the number of CP arrays does not match the size of the demultiplexing matrix.
    """
    if demultiplexing_matrix.shape[0] != demultiplexing_matrix.shape[1]:
        msg = "Demultiplexing matrix must be square."
        raise ValueError(msg)
    if demultiplexing_matrix.shape[0] != len(cp_arrays):
        msg = "number of CP arrays must match the size of the demultiplexing matrix."
        raise ValueError(msg)

    stacked = jnp.stack([jnp.asarray(a) for a in cp_arrays], axis=0)
    demultiplexing_matrix = jnp.asarray(demultiplexing_matrix)

    demultiplexed_arrays = jnp.tensordot(
        demultiplexing_matrix,
        stacked,
        axes=(1, 0),
    )

    return [demultiplexed_arrays[i] for i in range(demultiplexed_arrays.shape[0])]
