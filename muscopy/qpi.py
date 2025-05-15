"""Quantitative Phase Imaging (QPI) and MIP-QPI.

This module provides:

- `qpi`: A function to calculate the QPI phase image.
- `mip_qpi`: A function to calculate the MIP-QPI phase image.
- `correct_phase_offset`: A function to correct the phase offset of the QPI image.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

from muscopy.dh import get_spectrum, offaxis_dh

if TYPE_CHECKING:
    import types
    from collections.abc import Iterable

    from muscopy.backend_manager import ArrayProtocol
    from muscopy.cfg import OffsetRegions, Region
    from muscopy.dh import MuParameters


@typing.overload
def qpi(
    backend: types.ModuleType,
    array: ArrayProtocol,
    reference: ArrayProtocol,
    params: MuParameters,
    offaxis_centers: tuple[int, int],
) -> ArrayProtocol: ...


@typing.overload
def qpi(
    backend: types.ModuleType,
    array: ArrayProtocol,
    reference: ArrayProtocol,
    params: MuParameters,
    offaxis_centers: Iterable[tuple[int, int]],
) -> list[ArrayProtocol]: ...


def qpi(
    backend: types.ModuleType,
    array: ArrayProtocol,
    reference: ArrayProtocol,
    params: MuParameters,
    offaxis_centers: tuple[int, int] | Iterable[tuple[int, int]],
) -> ArrayProtocol | list[ArrayProtocol]:
    r"""Calculate the QPI phase image.

    Parameters
    ----------
    backend : `types.ModuleType`
        numpy or cupy module
    array : `ArrayProtocol`
        Hologram array
    reference : `ArrayProtocol`
        Reference hologram array
    params : `MuParameters`
        Microscopy Parameters class
    offaxis_centers : `Iterable`\[`tuple`\[`int`, `int`\]\]
        The crop centers of off-axis digital holography

    Returns
    -------
    `list`\[`ArrayProtocol`\]
        The QPI phase image
    """
    cp_fields = offaxis_dh(backend, array, reference, params, offaxis_centers)

    if isinstance(cp_fields, list):
        return [backend.angle(cp_field) for cp_field in cp_fields]

    return backend.angle(cp_fields)


def mip_qpi(  # noqa: PLR0913
    backend: types.ModuleType,
    array_on: ArrayProtocol,
    array_off: ArrayProtocol,
    params: MuParameters,
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
    params : `MuParameters`
        Micorsocpy Parameters class
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

    ft_array_on = backend.fft.fftshift(backend.fft.fft2(array_on)) * params.hologram2spectrum
    ft_array_off = backend.fft.fftshift(backend.fft.fft2(array_off)) * params.hologram2spectrum

    spectrum_on = get_spectrum(backend, ft_array_on, params, offaxis_center, crop_center=crop_center, c_r=c_r)
    spectrum_off = get_spectrum(backend, ft_array_off, params, offaxis_center, crop_center=crop_center, c_r=c_r)

    cp_field_on = backend.fft.ifft2(backend.fft.ifftshift(spectrum_on)) * params.spectrum2cpfield
    cp_field_off = backend.fft.ifft2(backend.fft.ifftshift(spectrum_off)) * params.spectrum2cpfield

    array_div = cp_field_on / cp_field_off

    if mip_center_reg is not None:
        center_phase = backend.mean(
            backend.angle(
                array_div[
                    mip_center_reg[0][0] : mip_center_reg[0][1],
                    mip_center_reg[1][0] : mip_center_reg[1][1],
                ],
            ),
        )
        if center_phase < 0:
            array_div = 1 / array_div

    return backend.angle(array_div)


def correct_phase_offset(
    backend: types.ModuleType,
    phase_array: ArrayProtocol,
    offset_regs: OffsetRegions,
) -> ArrayProtocol:
    """Correct the phase offset of the phase array.

    Parameters
    ----------
    backend : types.ModuleType
        numpy or cupy module
    phase_array : ArrayProtocol
        Phase array to be corrected
    offset_regs : OffsetRegions
        The regions to be used for phase offset correction

    Returns
    -------
    ArrayProtocol
        The phase array with the offset corrected
    """
    if offset_regs is None:
        return phase_array
    phase_offset_list = [
        backend.mean(phase_array[region[0][0] : region[0][1], region[1][0] : region[1][1]]) for region in offset_regs
    ]
    phase_offset = backend.mean(backend.array(phase_offset_list))

    return phase_array - phase_offset
