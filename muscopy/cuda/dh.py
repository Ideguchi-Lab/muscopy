"""CuPy implementation of digital holography helpers."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import cupy as cp

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from muscopy.dh import MuParameters

CENTER_COORDINATES = 2

__all__ = [
    "correct_aberration",
    "crop_array",
    "get_spectrum",
    "get_spectrums",
    "make_disk",
    "offaxis_dh",
]


def make_disk(
    center: tuple[float, float],
    radius: float,
    array_shape: int | tuple[int, int],
    *,
    highpass: bool = False,
) -> cp.ndarray:
    r"""Make a CuPy disk mask with specified center and radius.

    Parameters
    ----------
    center : `tuple`\[`float`, `float`\]
        The center position of the disk mask.
    radius : `float`
        The radius of the disk mask.
    array_shape : `int` | `tuple`\[`int`, `int`\]
        The shape of the array. If int, it is assumed to be square.
    highpass : `bool`, optional
        Apply highpass mask or not, by default False.

    Returns
    -------
    `cupy.ndarray`
        The disk mask with specified center and radius.
    """
    return _make_disk(center, radius, array_shape, highpass=highpass)


def crop_array(array: cp.ndarray, center: tuple[int, int], width: int) -> cp.ndarray:
    r"""Crop a CuPy array to the specified odd width around the center.

    Parameters
    ----------
    array : `cupy.ndarray`
        The array to be cropped.
    center : `tuple`\[`int`, `int`\]
        The center position of the crop.
    width : `int`
        The odd width of the center-symmetric crop.

    Returns
    -------
    `cupy.ndarray`
        The cropped array.

    """
    return _crop_array(array, center, width)


def get_spectrum(
    ft_array: cp.ndarray,
    params: MuParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> cp.ndarray:
    r"""Get a cropped CuPy spectrum from a hologram Fourier spectrum.

    Parameters
    ----------
    ft_array : `cupy.ndarray`
        Fourier spectrum of the hologram array.
    params : `MuParameters`
        Microscopy parameters.
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of off-axis digital holography.
    crop_center : `bool`, optional
        Whether to remove the center component, by default False.
    c_r : `int`, optional
        Radius of the center high-pass mask used when ``crop_center`` is True,
        by default 5.

    Returns
    -------
    `cupy.ndarray`
        The spectrum of complex amplitude.
    """
    return _get_spectrum(ft_array, params, offaxis_center, crop_center=crop_center, c_r=c_r)


def get_spectrums(
    ft_array: cp.ndarray,
    params: MuParameters,
    offaxis_centers: Iterable[tuple[int, int]],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> list[cp.ndarray]:
    r"""Get multiple cropped CuPy spectrums from a hologram Fourier spectrum.

    Parameters
    ----------
    ft_array : `cupy.ndarray`
        Fourier transformed hologram array.
    params : `MuParameters`
        Microscopy parameters.
    offaxis_centers : `collections.abc.Iterable`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography.
    crop_center : `bool`, optional
        Whether to remove the center component, by default False.
    c_r : `int`, optional
        Radius of the center high-pass mask used when ``crop_center`` is True,
        by default 5.

    Returns
    -------
    `list`\[`cupy.ndarray`\]
        The spectrums of complex amplitude.
    """
    return [
        _get_spectrum(ft_array, params, offaxis_center, crop_center=crop_center, c_r=c_r)
        for offaxis_center in offaxis_centers
    ]


def correct_aberration(spectrum: cp.ndarray, pupil_func: cp.ndarray) -> cp.ndarray:
    """Correct a CuPy spectrum aberration using the pupil function.

    Parameters
    ----------
    spectrum : `cupy.ndarray`
        The input spectrum to be corrected.
    pupil_func : `cupy.ndarray`
        The pupil function used for correction.

    Returns
    -------
    `cupy.ndarray`
        The corrected spectrum.

    """
    return _correct_aberration(spectrum, pupil_func)


@typing.overload
def offaxis_dh(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: tuple[int, int],
    *,
    pupil_func: cp.ndarray | None = None,
) -> cp.ndarray: ...


@typing.overload
def offaxis_dh(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: Sequence[tuple[int, int]],
    *,
    pupil_func: cp.ndarray | None = None,
) -> list[cp.ndarray]: ...


def offaxis_dh(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: tuple[int, int] | Sequence[tuple[int, int]],
    *,
    pupil_func: cp.ndarray | None = None,
) -> cp.ndarray | list[cp.ndarray]:
    r"""Reconstruct a complex wave front using CuPy off-axis digital holography.

    Parameters
    ----------
    array : `cupy.ndarray`
        Hologram array.
    reference : `cupy.ndarray`
        Reference hologram array.
    params : `MuParameters`
        Microscopy parameters.
    offaxis_centers : `tuple`\[`int`, `int`\] | `collections.abc.Sequence`\[`tuple`\[`int`, `int`\]\]
        The crop center or crop centers of off-axis digital holography.
    pupil_func : `cupy.ndarray` | `None`, optional
        Pupil function for aberration correction, by default None.

    Returns
    -------
    `cupy.ndarray` | `list`\[`cupy.ndarray`\]
        The complex wave front. Returns a single array when ``offaxis_centers``
        is a single center tuple, or a list of arrays when it is a sequence of
        center tuples.

    """
    return _offaxis_dh(array, reference, params, offaxis_centers, pupil_func=pupil_func)


def _as_single_center(
    offaxis_centers: tuple[int, int] | Sequence[tuple[int, int]],
) -> tuple[int, int] | None:
    if len(offaxis_centers) != CENTER_COORDINATES:
        return None

    first = offaxis_centers[0]
    second = offaxis_centers[1]
    if isinstance(first, int) and isinstance(second, int):
        return (first, second)

    return None


def _validate_center(center: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(center, tuple) and len(center) == CENTER_COORDINATES:
        return center

    msg = "offaxis_centers must be a center tuple or a sequence of center tuples"
    raise TypeError(msg)


def _offaxis_dh(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: tuple[int, int] | Sequence[tuple[int, int]],
    *,
    pupil_func: cp.ndarray | None = None,
) -> cp.ndarray | list[cp.ndarray]:
    params.verify_parameters()
    if array.shape != reference.shape:
        msg = "Array and reference must have the same shape"
        raise ValueError(msg)

    ft_array = cp.fft.fftshift(cp.fft.fft2(array)) * params.hologram2spectrum
    ft_reference = cp.fft.fftshift(cp.fft.fft2(reference)) * params.hologram2spectrum
    single_center = _as_single_center(offaxis_centers)
    if single_center is not None:
        spectrum = _get_spectrum(ft_array, params, single_center)
        ref_spectrum = _get_spectrum(ft_reference, params, single_center)

        if pupil_func is not None:
            spectrum = _correct_aberration(spectrum, pupil_func)
            ref_spectrum = _correct_aberration(ref_spectrum, pupil_func)

        cp_field = cp.fft.ifft2(cp.fft.ifftshift(spectrum)) * params.spectrum2cpfield
        ref_cp_field = cp.fft.ifft2(cp.fft.ifftshift(ref_spectrum)) * params.spectrum2cpfield
        cp_field /= ref_cp_field
        return cp_field

    cp_fields = []
    for offaxis_center_item in offaxis_centers:
        offaxis_center = _validate_center(offaxis_center_item)
        spectrum = _get_spectrum(ft_array, params, offaxis_center)
        ref_spectrum = _get_spectrum(ft_reference, params, offaxis_center)

        if pupil_func is not None:
            spectrum = _correct_aberration(spectrum, pupil_func)
            ref_spectrum = _correct_aberration(ref_spectrum, pupil_func)

        cp_field = cp.fft.ifft2(cp.fft.ifftshift(spectrum)) * params.spectrum2cpfield
        ref_cp_field = cp.fft.ifft2(cp.fft.ifftshift(ref_spectrum)) * params.spectrum2cpfield
        cp_field /= ref_cp_field
        cp_fields.append(cp_field)

    return cp_fields


def _make_disk(
    center: tuple[float, float],
    radius: float,
    array_shape: int | tuple[int, int],
    *,
    highpass: bool = False,
) -> cp.ndarray:
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = cp.meshgrid(cp.arange(array_shape[0]), cp.arange(array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    return circle > radius**2 if highpass else circle <= radius**2


def _crop_array(array: cp.ndarray, center: tuple[int, int], width: int) -> cp.ndarray:
    if width <= 0 or width % 2 == 0:
        msg = "width must be a positive odd integer"
        raise ValueError(msg)

    return array[
        center[0] - width // 2 : center[0] + width // 2 + 1,
        center[1] - width // 2 : center[1] + width // 2 + 1,
    ]


def _get_spectrum(
    ft_array: cp.ndarray,
    params: MuParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
) -> cp.ndarray:
    mask = _make_disk(offaxis_center, params.aperturesize_px // 2, params.img_size_px)
    masked_ft_array = ft_array * mask

    if crop_center:
        mask_highpass = _make_disk(offaxis_center, c_r, params.img_size_px, highpass=True)
        masked_ft_array *= mask_highpass

    return _crop_array(masked_ft_array, offaxis_center, params.aperturesize_px)


def _correct_aberration(spectrum: cp.ndarray, pupil_func: cp.ndarray) -> cp.ndarray:
    if spectrum.shape != pupil_func.shape:
        msg = f"Spectrum shape {spectrum.shape} and pupil function shape {pupil_func.shape} must match"
        raise ValueError(msg)
    return spectrum / pupil_func
