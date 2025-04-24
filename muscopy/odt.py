"""Optical Diffraction Tomography (ODT) calculator."""

from __future__ import annotations

from types import ModuleType
from dataclasses import dataclass
from functools import cached_property
from collections.abc import Sequence, Iterable

from tqdm import tqdm

from muscopy.backend_manager import ArrayProtocol
from muscopy.cfg import ArrayPrecision
from muscopy.qpi import QPIParameters, make_disk, correct_offset, crop_array
from muscopy.qpi_utils import unwrap_phase


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


def synthesize_spectrum(
    backend: ModuleType,
    scattering_wave_spectrums: Iterable[ArrayPrecision],
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

    for scattering_wave_spectrum in scattering_wave_spectrums:
        kz_disk = calc_kz_disk(backend, params, illumination_vector, precision)
        scattering_potential = _embed_3d_spectrum(
            backend, scattering_wave_spectrum * 2j * kz_disk, params, illumination_vector, precision, mode=mode
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
        scattering_spectrum = _calc_1st_scattering_spectrum(backend, cp_field, ref_cp_field, config.approx_type)
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


def _calc_1st_scattering_spectrum() -> ArrayProtocol:
    pass


def _embed_3d_spectrum() -> ArrayProtocol:
    pass
