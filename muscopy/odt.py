"""Optical Diffraction Tomography (ODT) calculator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

from tqdm import tqdm

from muscopy.cfg import ArrayPrecision
from muscopy.qpi import QPIParameters, make_disk
from muscopy.qpi_utils import unwrap_phase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from types import ModuleType

    from muscopy.backend_manager import ArrayProtocol


@dataclass
class ODTParameters(QPIParameters):
    na_illumination: float = 1.0

    def verify_parameters(self) -> None:
        """Verify the parameters.

        Raises
        ------
        ValueError
            1. If the NA of illumination is negative.
            2. If the NA of illumination is greater than the NA of the objective.
            3. If the NA of illumination is greater than the NA of the solvent.
        """
        super().verify_parameters()
        if self.na_illumination < 0:
            msg = "NA of illumination cannot be negative."
            raise ValueError(msg)
        if self.na_illumination > self.na:
            msg = "NA of illumination cannot be greater than NA of the objective."
            raise ValueError(msg)
        if self.na_illumination > self.n_sol:
            msg = "NA of illumination cannot be greater than NA of the solvent."
            raise ValueError(msg)

    @cached_property
    def light_freq_lateral_px(self) -> float:
        """Calculate the lateral frequency in pixels.

        Returns
        -------
        `float`
            The lateral frequency in pixels.
        """
        return self.light_freq_px * self.na_illumination / self.n_sol

    @cached_property
    def light_freq_axial_px(self) -> float:
        """Calculate the axial frequency in pixels.

        Returns
        -------
        `float`
            The axial frequency in pixels.
        """
        return (self.light_freq_px**2 - self.light_freq_lateral_px**2) ** 0.5

    @cached_property
    def freq_axial_extent_px(self) -> int:
        """Calculate the axial extent of Fourier space in pixels.

        Returns
        -------
        `int`
            The axial extent of Fourier space in pixels.
        """
        return 2 * int(self.light_freq_px) + 1

    @cached_property
    def imgpx_axial_m_per_px(self) -> float:
        """Calculate the axial pixel size in meters.

        Returns
        -------
        `float`
            The axial pixel size in meters.
        """
        return 1 / (self.freq_per_px * self.freq_axial_extent_px)

    @cached_property
    def spectrum2field_z(self) -> float:
        """Axial Fourier factor from spectrum to field.

        Returns
        -------
        `float`
            factor from spectrum to field.
        """
        return (self.freq_per_px / self.imgpx_axial_m_per_px) ** 0.5

    @cached_property
    def field_z2spectrum(self) -> float:
        """Axial Fourier factor from spectrum to field.

        Returns
        -------
        `float`
            factor from field to spectrum.
        """
        return (self.imgpx_axial_m_per_px / self.freq_per_px) ** 0.5


@dataclass
class ODTConfig:
    approx_type: str = "Born"
    hermite_symmetry: bool = True
    precision: ArrayPrecision = ArrayPrecision()
    edge_size: int = 0


class ScatteringSpectrum(NamedTuple):
    array: ArrayProtocol
    illumintaion_vector: tuple[int, int]


def synthesize_spectrum(
    backend: ModuleType,
    scattering_spectrums: Iterable[ScatteringSpectrum],
    params: ODTParameters,
    config: ODTConfig,
    mode: str = "Forward",
) -> ArrayProtocol:
    synthesized_spectrum = backend.zeros(
        (
            2 * params.aperturesize_px + 1 - config.edge_size,
            2 * params.aperturesize_px + 1 - config.edge_size,
            2 * params.freq_axial_extent_px + 1,
        ),
        dtype=config.precision.get_complex_precision(),
    )
    synthesized_weight = backend.ones_like(synthesized_spectrum, dtype=config.precision.get_int_precision())

    print("Synthesize spectrum...")  # noqa: T201
    for scattering_spectrum in tqdm(scattering_spectrums):
        kz_disk = _calc_kz_disk(
            backend, params, scattering_spectrum.array.shape, scattering_spectrum.illumination_vector, config.precision
        )
        scattering_potential = _embed_3d_spectrum(
            backend,
            scattering_spectrum.array * 2j * kz_disk,
            synthesized_spectrum.shape,
            params,
            scattering_spectrum.illumination_vector,
            mode=mode,
        )

        synthesized_spectrum += scattering_potential
        synthesized_weight += scattering_potential != 0

    synthesized_weight -= synthesized_weight > 1
    synthesized_spectrum /= synthesized_weight

    # NOTE: Fourier factor to cancel 3D mapping effect

    return synthesized_spectrum


def fill_hermite_components(backend: ModuleType, spectrum3d: ArrayProtocol) -> ArrayProtocol:
    conjugate_spectrum = backend.conjugate(backend.flip(spectrum3d, axis=(0, 1, 2)))
    overlap_region = backend.logical_and(backend.abs(spectrum3d) > 0, backend.abs(conjugate_spectrum) > 0)
    spectrum3d += conjugate_spectrum
    spectrum3d[overlap_region] /= 2
    return spectrum3d


def calc_refractive_index(
    backend: ModuleType, scattering_potential: ArrayProtocol, params: ODTParameters
) -> ArrayProtocol:
    return params.n_sol * backend.sqrt(
        backend.ones_like(scattering_potential)
        + scattering_potential / (params.light_freq_px * params.freq_per_px) ** 2
    )


def odt(
    backend: ModuleType,
    cp_fields: Sequence[ArrayProtocol],
    ref_cp_fields: Sequence[ArrayProtocol],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[ArrayProtocol, ArrayProtocol]:
    # weak scattering approximation
    scattering_spectrums = []
    for cp_field, ref_cp_field in zip(cp_fields, ref_cp_fields):
        cp_spectrum = backend.fft.fftshift(backend.fft.fft2(cp_field))
        max_x, max_y, _ = _find_max_args(backend, cp_spectrum)
        illumination_vector = (max_x - params.img_center[0], max_y - params.img_center[1])
        expanded_cp_field = _shift_dh_spectrum(backend, params, cp_field, illumination_vector)
        expanded_ref_cp_field = _shift_dh_spectrum(backend, params, ref_cp_field, illumination_vector)
        scattering_spectrum_array = _calc_1st_scattering_spectrum(
            backend, expanded_cp_field, expanded_ref_cp_field, params, config.approx_type, illumination_vector
        )
        scattering_spectrum = ScatteringSpectrum(scattering_spectrum_array, illumination_vector)
        scattering_spectrums.append(scattering_spectrum)

    synthesized_spectrum = synthesize_spectrum(backend, scattering_spectrums, params, config, mode="Forward")

    if config.hermite_symmetry:
        synthesized_spectrum = fill_hermite_components(backend, synthesized_spectrum)

    scattering_potential = backend.fft.ifftn(backend.fft.ifftshift(synthesized_spectrum))

    refractive_index = calc_refractive_index(backend, scattering_potential, params)
    return refractive_index, scattering_potential


def _find_max_args(backend: ModuleType, array: ArrayProtocol) -> tuple[int, int, float]:
    max_value = backend.max(array)
    max_index = backend.unravel_index(backend.argmax(array), array.shape)
    max_x = max_index[0]
    max_y = max_index[1]
    return max_x, max_y, max_value


def _shift_dh_spectrum(
    backend: ModuleType, params: ODTParameters, cp_field: ArrayProtocol, illumination_vector: tuple[int, int]
) -> ArrayProtocol:
    expanded_cp_field = backend.zeros(
        (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1), dtype=cp_field.dtype
    )

    expanded_cp_field[
        params.aperturesize_px // 2 - illumination_vector[0] : 3 * params.aperturesize_px // 2 - illumination_vector[0],
        params.aperturesize_px // 2 - illumination_vector[1] : 3 * params.aperturesize_px // 2 - illumination_vector[1],
    ] = cp_field
    return expanded_cp_field


def _calc_1st_scattering_spectrum(
    backend: ModuleType,
    cp_field: ArrayProtocol,
    ref_cp_field: ArrayProtocol,
    params: ODTParameters,
    approx_type: str,
    illumination_vector: tuple[int, int],
) -> ArrayProtocol:
    if approx_type == "Born":
        scattering_field = (cp_field - ref_cp_field) / ref_cp_field
    elif approx_type == "Rytov":
        scattering_field = _log_field(backend, cp_field, ref_cp_field)
    else:
        msg = f"Unknown approximation type: {approx_type}"
        raise ValueError(msg)
    scattering_spectrum = backend.fft.fftshift(backend.fft.fft2(scattering_field))
    mask_for_synthesis = make_disk(
        backend,
        (
            params.aperturesize_px + illumination_vector[0],
            params.aperturesize_px + illumination_vector[1],
        ),
        params.aperturesize_px // 2,
        cp_field.shape,
    )
    scattering_spectrum *= mask_for_synthesis

    return scattering_spectrum


def _log_field(backend: ModuleType, cp_field: ArrayProtocol, ref_cp_field: ArrayProtocol) -> ArrayProtocol:
    field_log = backend.log(cp_field)
    field_log_real = backend.real(field_log)
    field_log_imag = backend.imag(field_log)
    ref_field_log = backend.log(ref_cp_field)
    ref_field_log_real = backend.real(ref_field_log)
    ref_field_log_imag = backend.imag(ref_field_log)

    amplitude = field_log_real - ref_field_log_real
    phase = unwrap_phase(bmg, field_log_imag - ref_field_log_imag)

    return amplitude + 1j * phase


def _embed_3d_spectrum(
    backend: ModuleType,
    spectrum2d: ArrayProtocol,
    shape_3d: tuple[int, int, int],
    params: ODTParameters,
    illumination_vector: tuple[int, int],
    mode: str,
) -> ArrayProtocol:
    xx, yy = backend.meshgrid(
        backend.arange(-shape_3d[0] // 2, shape_3d[0] // 2),
        backend.arange(
            -shape_3d[1] // 2,
            shape_3d[1] // 2,
        ),
        indexing="ij",
    )

    _, _, zz = backend.meshgrid(
        backend.arange(-shape_3d[0] // 2, shape_3d[0] // 2),
        backend.arange(-shape_3d[1] // 2, shape_3d[1] // 2),
        backend.arange(-shape_3d[2]) // 2,
        shape_3d[2] // 2,
        indexing="ij",
    )

    circle = (xx + illumination_vector[0]) ** 2 + (yy + illumination_vector[1]) ** 2 <= (
        params.aperturesize_px // 2
    ) ** 2

    fz_circle = (
        backend.sqrt(params.light_freq_px**2 - (xx + illumination_vector[0]) ** 2 - (yy + illumination_vector[1]) ** 2)
        - params.light_freq_axial_px
    )

    if mode == "Backward":
        fz_circle = -fz_circle

    fz_value = (fz_circle + shape_3d[2] // 2) * circle
    fz_tile = backend.tile(fz_value, (shape_3d[2], 1, 1))
    fz_tile = fz_tile.transpose(1, 2, 0)

    fz_tile -= fz_tile == 0
    fz_index = zz == fz_tile

    array_tiled = backend.stack([spectrum2d] * shape_3d[2], axis=2)
    return array_tiled * fz_index


def _calc_kz_disk(
    backend: ModuleType,
    params: ODTParameters,
    shape: tuple[int, int],
    illumination_vector: tuple[int, int],
    precision: ArrayPrecision,
) -> ArrayProtocol:
    xx, yy = backend.measgrid(
        backend.arange(-shape[0] // 2, shape[0] // 2),
        backend.arange(-shape[1] // 2, shape[1] // 2),
        indexing="ij",
    )

    disk = (xx + illumination_vector[0]) ** 2 + (yy + illumination_vector[1]) ** 2
    disk_mask = disk <= (params.aperturesize_px // 2) ** 2
    fz_disk = (params.light_freq_px**2 - disk) ** 0.5 * disk_mask
    kz_disk = fz_disk * params.k_per_px
    return kz_disk.astype(precision.get_float_precision())
