"""CuPy implementation of Quantitative Phase Imaging (QPI) and MIP-QPI."""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import cupy as cp

from muscopy.cuda.dh import get_spectrum, offaxis_dh

if TYPE_CHECKING:
    from collections.abc import Sequence

    from muscopy.cfg import OffsetRegions, Region
    from muscopy.dh import MuParameters

__all__ = [
    "correct_phase_offset",
    "mip_qpi",
    "qpi",
]


@typing.overload
def qpi(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: tuple[int, int],
    *,
    pupil_func: cp.ndarray | None = None,
) -> cp.ndarray: ...


@typing.overload
def qpi(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: Sequence[tuple[int, int]],
    *,
    pupil_func: cp.ndarray | None = None,
) -> list[cp.ndarray]: ...


def qpi(
    array: cp.ndarray,
    reference: cp.ndarray,
    params: MuParameters,
    offaxis_centers: tuple[int, int] | Sequence[tuple[int, int]],
    *,
    pupil_func: cp.ndarray | None = None,
) -> cp.ndarray | list[cp.ndarray]:
    r"""Calculate a QPI phase image using CuPy.

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
        The QPI phase image. Returns a single array when ``offaxis_centers``
        is a single center tuple, or a list of arrays when it is a sequence of
        center tuples.
    """
    cp_fields = offaxis_dh(array, reference, params, offaxis_centers, pupil_func=pupil_func)

    if isinstance(cp_fields, list):
        return [cp.angle(cp_field) for cp_field in cp_fields]

    return cp.angle(cp_fields)


def mip_qpi(
    array_on: cp.ndarray,
    array_off: cp.ndarray,
    params: MuParameters,
    offaxis_center: tuple[int, int],
    *,
    crop_center: bool = False,
    c_r: int = 5,
    mip_center_reg: Region | None = None,
) -> cp.ndarray:
    r"""Calculate a MIP-QPI phase image using CuPy.

    Parameters
    ----------
    array_on : `cupy.ndarray`
        MIR ON hologram array.
    array_off : `cupy.ndarray`
        MIR OFF hologram array.
    params : `MuParameters`
        Microscopy parameters.
    offaxis_center : `tuple`\[`int`, `int`\]
        The crop center of off-axis digital holography.
    crop_center : `bool`, optional
        Whether to remove the low-frequency center component from each cropped
        off-axis spectrum before reconstruction, by default False.
    c_r : `int`, optional
        Radius of the center high-pass mask used when ``crop_center`` is True,
        by default 5.
    mip_center_reg : `Region` | `None`, optional
        Region used to estimate the central phase sign. When provided and the
        mean phase in this region is negative, the MIR ON/OFF field ratio is
        inverted before taking the output phase, by default None.

    Returns
    -------
    `cupy.ndarray`
        The MIP-QPI phase image computed as the angle of the MIR ON/OFF field ratio.

    Raises
    ------
    ValueError
        If the array on and off have different shapes.
    """
    if array_on.shape != array_off.shape:
        msg = "Array on and off must have the same shape"
        raise ValueError(msg)

    ft_array_on = cp.fft.fftshift(cp.fft.fft2(array_on)) * params.hologram2spectrum
    ft_array_off = cp.fft.fftshift(cp.fft.fft2(array_off)) * params.hologram2spectrum

    spectrum_on = get_spectrum(ft_array_on, params, offaxis_center, crop_center=crop_center, c_r=c_r)
    spectrum_off = get_spectrum(ft_array_off, params, offaxis_center, crop_center=crop_center, c_r=c_r)

    cp_field_on = cp.fft.ifft2(cp.fft.ifftshift(spectrum_on)) * params.spectrum2cpfield
    cp_field_off = cp.fft.ifft2(cp.fft.ifftshift(spectrum_off)) * params.spectrum2cpfield

    array_div = cp_field_on / cp_field_off

    if mip_center_reg is not None:
        center_phase = cp.mean(
            cp.angle(
                array_div[
                    mip_center_reg[0][0] : mip_center_reg[0][1],
                    mip_center_reg[1][0] : mip_center_reg[1][1],
                ],
            ),
        )
        if float(center_phase) < 0:
            array_div = 1 / array_div

    return cp.angle(array_div)


def correct_phase_offset(
    phase_array: cp.ndarray,
    offset_regs: OffsetRegions,
) -> cp.ndarray:
    """Correct the phase offset of a CuPy phase array.

    Parameters
    ----------
    phase_array : `cupy.ndarray`
        Phase array to be corrected.
    offset_regs : `OffsetRegions`
        The regions to be used for phase offset correction.

    Returns
    -------
    `cupy.ndarray`
        The phase array with the offset corrected.
    """
    if not offset_regs:
        return phase_array

    phase_offset_list = [
        cp.mean(phase_array[region[0][0] : region[0][1], region[1][0] : region[1][1]]) for region in offset_regs
    ]
    phase_offset = cp.mean(cp.asarray(phase_offset_list))

    return phase_array - phase_offset
