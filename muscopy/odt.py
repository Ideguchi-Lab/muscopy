"""Optical Diffraction Tomography (ODT) calculator.

This module provides:

- `ODTParameters`: A class to store ODT parameters.
- `ODTConfig`: A class to store ODT configuration.
- `ScatteringSpectrum`: A class to store scattering spectrum.
- `synthesize_spectrum`: A function to synthesize scattering spectrums into 3D scattering potential.
- `fill_hermite_components`: A function to fill the hermite conjugated spectrum for transparent sample.
- `calc_refractive_index`: A function to calculate the refractive index from the scattering potential.
- `calc_scattering_spectrums`: A function to calculate first-order scattering spectrums.
- `calc_scattering_potential_from_spectrums`: A function to reconstruct scattering potential from spectrums.
- `calc_scattering_potential`: A function to calculate the scattering potential from complex field spectrums.
- `odt`: A function to perform ODT reconstruction.
- `calculate_odt_difference`: A function to calculate the difference between two ODT reconstructions.
- `discard_higher_axial_freq`: A function to discard higher axial frequency.
- `zeropad_higher_axial_freq`: A function to zero pad the higher axial frequency.
"""

from __future__ import annotations

import dataclasses
import warnings
from enum import StrEnum
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from muscopy.cfg import ArrayPrecision, OffsetRegions
from muscopy.dh import MuParameters, correct_gradient, correct_offset, make_disk
from muscopy.qpi_utils import unwrap_phase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


EPSILON = 1e-8
_ODT_DEPRECATION_REMOVAL_VERSION = "0.9.0"


class EwaldEmbeddingMode(StrEnum):
    """Ewald sphere embedding mode."""

    TRUNCATE = "truncate"
    NEAREST = "nearest"
    LINEAR = "linear"


@dataclasses.dataclass
class ODTParameters(MuParameters):
    """Optical Diffraction Tomography (ODT) parameters.

    Attributes
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
    na_illumination : `float`
        Maximum illumination numerical aperture
    """

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

    @property
    def freq_axial_extent_px(self) -> int:
        """Calculate the axial extent of Fourier space in pixels.

        Returns
        -------
        `int`
            The axial extent of Fourier space in pixels.
        """
        return 2 * (self.aperturesize_px // 2) + 1

    @property
    def imgpx_lateral_m_per_px(self) -> float:
        """Calculate the lateral pixel size in meters (ODT).

        Returns
        -------
        `float`
            The lateral pixel size in meter
        """
        return self.px_size_m * self.img_size_px / (2 * self.aperturesize_px + 1)

    @property
    def imgpx_axial_m_per_px(self) -> float:
        """Calculate the axial pixel size in meters.

        Returns
        -------
        `float`
            The axial pixel size in meters.
        """
        return 1 / (self.freq_per_px * self.freq_axial_extent_px)

    @property
    def spectrum2cpfield_xy(self) -> float:
        """Lateral Fourier factor from spectrum to field.

        Returns
        -------
        `float`
            factor from spectrum to field
        """
        return float((self.freq_per_px / self.imgpx_lateral_m_per_px) ** 0.5)

    @property
    def cpfield_xy2spectrum(self) -> float:
        """Lateral Fourier factor from field to spectrum.

        Returns
        -------
        `float`
            factor from field to spectrum
        """
        return float((self.imgpx_lateral_m_per_px / self.freq_per_px) ** 0.5)

    @property
    def spectrum2cpfield_z(self) -> float:
        """Axial Fourier factor from spectrum to field.

        Returns
        -------
        `float`
            factor from spectrum to field.
        """
        return float((self.freq_per_px / self.imgpx_axial_m_per_px) ** 0.5)

    @property
    def cpfield_z2spectrum(self) -> float:
        """Axial Fourier factor from spectrum to field.

        Returns
        -------
        `float`
            factor from field to spectrum.
        """
        return float((self.imgpx_axial_m_per_px / self.freq_per_px) ** 0.5)


@dataclasses.dataclass
class ODTConfig:
    """Optical Diffraction Tomography (ODT) configuration.

    Attributes
    ----------
    approx_type : `str`, optional
        Weak scattering approximation type, [Born, Rytov].
    linear_approx : `bool`, optional
        Whether to use linear approximation or not.
        The approximation linearizes scattering potential to the refractive index distribution.
    hermite_symmetry : `bool`, optional
        Whether to use Hermite embedding or not.
    precision : `ArrayPrecision`
        Precision configuration
    edge_size : `int`, optional
        Size of removed edge in FT calculation.
    offset_regions : `OffsetRegions`, optional
        Regions to be used for offset calculation.
    gradient_correction : `bool`, optional
        Whether to remove a median-estimated linear phase ramp from each scattering field.
    ewald_embedding_mode : `EwaldEmbeddingMode`, optional
        Ewald sphere embedding mode. ``EwaldEmbeddingMode.TRUNCATE`` preserves the
        legacy integer truncation, ``EwaldEmbeddingMode.NEAREST`` places each sample
        on the nearest axial plane, and ``EwaldEmbeddingMode.LINEAR`` splats each
        sample into adjacent axial planes with linear weights.
    verbose : `bool`, optional
        Whether to print reconstruction status and progress output.
    """

    approx_type: str = "Born"
    linear_approx: bool = False
    hermite_symmetry: bool = True
    precision: ArrayPrecision = dataclasses.field(default_factory=ArrayPrecision)
    edge_size: int = 0
    offset_regions: OffsetRegions = None
    gradient_correction: bool = True
    ewald_embedding_mode: EwaldEmbeddingMode = EwaldEmbeddingMode.TRUNCATE
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate enum-only configuration fields.

        Raises
        ------
        TypeError
            If ``ewald_embedding_mode`` is not an ``EwaldEmbeddingMode``.
        """
        if not isinstance(self.ewald_embedding_mode, EwaldEmbeddingMode):
            msg = "ewald_embedding_mode must be an EwaldEmbeddingMode."
            raise TypeError(msg)


@dataclasses.dataclass
class ScatteringSpectrum:
    r"""Data class to store scattering spectrum.

    Attributes
    ----------
    array : `Array`
        Scattering spectrum
    illumination_vector : `tuple`\[`int`, `int`\]
        Illumination vector in px unit
    coefficient : `complex`, optional
        Linear coefficient applied to this spectrum during ODT synthesis.
    """

    array: Array
    illumination_vector: tuple[int, int]
    coefficient: complex = 1.0 + 0.0j


def synthesize_spectrum(
    scattering_spectrums: Iterable[ScatteringSpectrum],
    params: ODTParameters,
    config: ODTConfig,
    mode: str = "Forward",
) -> Array:
    r"""Synthesize scattering spectrums into 3D scattering potential.

    Parameters
    ----------
    scattering_spectrums : `collections.abc.Iterable`\[`ScatteringSpectrum`\]
        Collection of ScatteringSpectrum object
    params : `ODTParameters`
        ODT params class instance
    config : `ODTConfig`
        ODT configuration
    mode : `str`, optional
        Diffraction mode, by default "Forward"

    Returns
    -------
    `Array`
        Synthesized scattering potential
    """
    synthesized_spectrum = jnp.zeros(
        (
            2 * params.aperturesize_px + 1,
            2 * params.aperturesize_px + 1,
            params.freq_axial_extent_px,
        ),
        dtype=config.precision.complex_precision(),
    )
    synthesized_weight = jnp.zeros_like(synthesized_spectrum, dtype=config.precision.float_precision())

    if config.verbose:
        print("Synthesize spectrum...")  # noqa: T201
    for scattering_spectrum in tqdm(scattering_spectrums, disable=not config.verbose):
        kz_disk = _calc_kz_disk(
            params, scattering_spectrum.array.shape[0], scattering_spectrum.illumination_vector, config.precision
        )
        scattering_potential, embedding_weight = _embed_3d_spectrum_with_weights(
            scattering_spectrum.array * 2j * kz_disk,
            (synthesized_spectrum.shape[0], synthesized_spectrum.shape[1], synthesized_spectrum.shape[2]),
            params,
            scattering_spectrum.illumination_vector,
            mode=mode,
            ewald_embedding_mode=config.ewald_embedding_mode,
        )

        synthesized_spectrum += scattering_spectrum.coefficient * scattering_potential
        synthesized_weight += embedding_weight

    synthesized_weight = jnp.where(synthesized_weight > 0, synthesized_weight, 1)
    return synthesized_spectrum / synthesized_weight


def fill_hermite_components(spectrum3d: Array) -> Array:
    """Fill the hermite conjugated spectrum for transparent sample.

    Parameters
    ----------
    spectrum3d : `Array`
        Spectrum of scattering potential

    Returns
    -------
    `Array`
        Filled spectrum of scattering potential
    """
    conjugate_spectrum = jnp.conjugate(jnp.flip(spectrum3d, axis=(0, 1, 2)))
    overlap_region = (jnp.abs(spectrum3d) > 0) & (jnp.abs(conjugate_spectrum) > 0)
    spectrum3d = spectrum3d + conjugate_spectrum  # noqa: PLR6104
    return jnp.where(overlap_region, spectrum3d / 2, spectrum3d)


def calc_refractive_index(scattering_potential: Array, params: ODTParameters) -> Array:
    """Calculate the refractive index from the scattering potential.

    Parameters
    ----------
    scattering_potential : `Array`
        scattering potential array
    params : `ODTParameters`
        ODT parameter instance

    Returns
    -------
    `Array`
        3D refractive index
    """
    return params.n_sol * jnp.sqrt(
        jnp.ones_like(scattering_potential) - scattering_potential / (params.light_freq_px * params.k_per_px) ** 2
    )


def calc_scattering_potential(
    cp_spectrums: Sequence[Array],
    ref_cp_spectrums: Sequence[Array],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[Array, Array]:
    r"""Calculate the scattering potential from complex field spectrums.

    .. deprecated:: 0.8.0
        Use :func:`calc_scattering_spectrums` followed by
        :func:`calc_scattering_potential_from_spectrums` instead.
        This wrapper is scheduled for removal in muscopy 0.9.0.

    Parameters
    ----------
    cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Spectrum of complex fields
    ref_cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Reference spectrum of complex fields
    params : `ODTParameters`
        ODT parameter instance
    config : `ODTConfig`
        ODT configuration

    Returns
    -------
    `tuple`\[`Array`, `Array`\]
        3D scattering potential, 3D spectrum
    """
    _warn_deprecated_odt_api(
        "calc_scattering_potential() is deprecated and will be removed in "
        f"muscopy {_ODT_DEPRECATION_REMOVAL_VERSION}. Use calc_scattering_spectrums() followed by "
        "calc_scattering_potential_from_spectrums() instead. Pass explicit illumination_vectors to "
        "calc_scattering_spectrums() when the illumination geometry is known."
    )
    scattering_spectrums = calc_scattering_spectrums(cp_spectrums, ref_cp_spectrums, params, config)
    return calc_scattering_potential_from_spectrums(scattering_spectrums, params, config)


def calc_scattering_spectrums(
    cp_spectrums: Sequence[Array],
    ref_cp_spectrums: Sequence[Array],
    params: ODTParameters,
    config: ODTConfig,
    illumination_vectors: Sequence[tuple[int, int]] | None = None,
) -> list[ScatteringSpectrum]:
    r"""Calculate first-order scattering spectrums from complex field spectrums.

    Parameters
    ----------
    cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Spectrum of complex fields.
    ref_cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Reference spectrum of complex fields.
    params : `ODTParameters`
        ODT parameter instance.
    config : `ODTConfig`
        ODT configuration.
    illumination_vectors : `collections.abc.Sequence`\[`tuple`\[`int`, `int`\]\] | `None`, optional
        Explicit illumination vectors in pixel units. If omitted, vectors are estimated
        from the reference spectrum peak as in the legacy path.

    Returns
    -------
    `list`\[`ScatteringSpectrum`\]
        First-order scattering spectrums with illumination metadata.

    Raises
    ------
    ValueError
        If explicit illumination vector count does not match the spectrum count.
    """
    params.verify_parameters()
    config.precision.validate()
    if len(cp_spectrums) != len(ref_cp_spectrums):
        msg = "cp_spectrums and ref_cp_spectrums must have the same length."
        raise ValueError(msg)
    if illumination_vectors is not None and len(illumination_vectors) != len(cp_spectrums):
        msg = "illumination_vectors must match the number of complex field spectrums."
        raise ValueError(msg)

    scattering_spectrums = []
    for index, (cp_spectrum, ref_cp_spectrum) in enumerate(zip(cp_spectrums, ref_cp_spectrums, strict=True)):
        if illumination_vectors is None:
            max_x, max_y, _ = _find_max_args(jnp.abs(ref_cp_spectrum))
            center_idx = cp_spectrum.shape[0] // 2
            illumination_vector = (max_x - center_idx, max_y - center_idx)
        else:
            illumination_vector = illumination_vectors[index]
        expanded_cp_spectrum = _shift_dh_spectrum(params, cp_spectrum, illumination_vector, config.edge_size)
        expanded_cp_spectrum = jnp.asarray(expanded_cp_spectrum, dtype=config.precision.complex_precision())
        expanded_ref_cp_spectrum = _shift_dh_spectrum(params, ref_cp_spectrum, illumination_vector, config.edge_size)
        expanded_ref_cp_spectrum = jnp.asarray(expanded_ref_cp_spectrum, dtype=config.precision.complex_precision())
        cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(expanded_cp_spectrum), norm="ortho")
        ref_cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(expanded_ref_cp_spectrum), norm="ortho")
        scattering_spectrum_array = _calc_1st_scattering_spectrum(
            cp_field,
            ref_cp_field,
            params,
            config.approx_type,
            illumination_vector,
            config.edge_size,
            config.offset_regions,
            config.gradient_correction,
        )
        scattering_spectrum = ScatteringSpectrum(scattering_spectrum_array, illumination_vector)
        scattering_spectrums.append(scattering_spectrum)

    return scattering_spectrums


def calc_scattering_potential_from_spectrums(
    scattering_spectrums: Iterable[ScatteringSpectrum],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[Array, Array]:
    r"""Calculate scattering potential from first-order scattering spectrums.

    Parameters
    ----------
    scattering_spectrums : `collections.abc.Iterable`\[`ScatteringSpectrum`\]
        First-order scattering spectrums with illumination metadata.
    params : `ODTParameters`
        ODT parameter instance.
    config : `ODTConfig`
        ODT configuration.

    Returns
    -------
    `tuple`\[`Array`, `Array`\]
        3D scattering potential and synthesized 3D spectrum.
    """
    params.verify_parameters()
    config.precision.validate()
    synthesized_spectrum = synthesize_spectrum(scattering_spectrums, params, config, mode="Forward")

    if config.hermite_symmetry:
        synthesized_spectrum = fill_hermite_components(synthesized_spectrum)

    scattering_potential = jnp.fft.ifftn(jnp.fft.ifftshift(synthesized_spectrum), norm="ortho")

    scattering_potential = jnp.fft.fftshift(scattering_potential, axes=(2))

    factor = params.spectrum2cpfield_xy**2 * params.spectrum2cpfield_z / (2 * jnp.pi) ** (3 / 2)
    scattering_potential *= factor

    return scattering_potential, synthesized_spectrum


def odt(
    cp_spectrums: Sequence[Array],
    ref_cp_spectrums: Sequence[Array],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[Array, Array]:
    r"""Optical Diffraction Tomography (ODT) reconstruction.

    Parameters
    ----------
    cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Spectrum of complex fields
    ref_cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Reference spectrum of complex fields
    params : `ODTParameters`
        ODT parameter instance
    config : `ODTConfig`
        ODT configuration

    Returns
    -------
    `tuple`\[`Array`, `Array`\]
        3D refractive index, 3D spectrum
    """
    scattering_spectrums = calc_scattering_spectrums(cp_spectrums, ref_cp_spectrums, params, config)
    scattering_potential, synthesized_spectrum = calc_scattering_potential_from_spectrums(
        scattering_spectrums,
        params,
        config,
    )
    if config.linear_approx:
        factor = -((params.light_freq_px * params.k_per_px) ** 2) * params.n_sol
        refractive_index = params.n_sol + scattering_potential / factor
    else:
        refractive_index = calc_refractive_index(scattering_potential, params)
    return refractive_index, synthesized_spectrum


def calculate_odt_difference(
    cp_spectrums_1: Sequence[Array],
    ref_cp_spectrums_1: Sequence[Array],
    cp_spectrums_2: Sequence[Array],
    ref_cp_spectrums_2: Sequence[Array],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[Array, Array, Array]:
    r"""Calculate the difference between two ODT reconstructions with same parameters.

    This function performs ODT reconstruction on two different datasets using identical
    parameters and returns the difference in refractive index and scattering potential.

    Parameters
    ----------
    cp_spectrums_1 : `collections.abc.Sequence`\[`Array`\]
        First dataset: spectrum of complex fields
    ref_cp_spectrums_1 : `collections.abc.Sequence`\[`Array`\]
        First dataset: reference spectrum of complex fields
    cp_spectrums_2 : `collections.abc.Sequence`\[`Array`\]
        Second dataset: spectrum of complex fields
    ref_cp_spectrums_2 : `collections.abc.Sequence`\[`Array`\]
        Second dataset: reference spectrum of complex fields
    params : `ODTParameters`
        ODT parameter instance (same for both datasets)
    config : `ODTConfig`
        ODT configuration (same for both datasets)

    Returns
    -------
    `tuple`\[`Array`, `Array`, `Array`\]
        Difference in refractive index (dataset1 - dataset2),
        refractive index from dataset1,
        refractive index from dataset2

    Raises
    ------
    ValueError
        If the input datasets have different numbers of spectrums or reference spectrums,
        or if the number of spectrums and reference spectrums don't match within each dataset
    """
    if len(cp_spectrums_1) != len(cp_spectrums_2):
        msg = "The number of spectrums in both datasets must be the same"
        raise ValueError(msg)
    if len(ref_cp_spectrums_1) != len(ref_cp_spectrums_2):
        msg = "The number of reference spectrums in both datasets must be the same"
        raise ValueError(msg)
    if len(cp_spectrums_1) != len(ref_cp_spectrums_1):
        msg = "The number of spectrums and reference spectrums must match in dataset 1"
        raise ValueError(msg)
    if len(cp_spectrums_2) != len(ref_cp_spectrums_2):
        msg = "The number of spectrums and reference spectrums must match in dataset 2"
        raise ValueError(msg)

    if config.verbose:
        print("Reconstructing ODT from dataset 1...")  # noqa: T201
    refractive_index_1, _ = odt(cp_spectrums_1, ref_cp_spectrums_1, params, config)

    if config.verbose:
        print("Reconstructing ODT from dataset 2...")  # noqa: T201
    refractive_index_2, _ = odt(cp_spectrums_2, ref_cp_spectrums_2, params, config)

    # Calculate the difference
    refractive_index_diff = refractive_index_1 - refractive_index_2

    return refractive_index_diff, refractive_index_1, refractive_index_2


def discard_higher_axial_freq(array3d: Array, threshold: int) -> Array:
    """Discard higher axial frequency.

    Parameters
    ----------
    array3d : `Array`
        3D array to be modulated
    threshold : `int`
        Number of pixels to be remained

    Returns
    -------
    `Array`
        3D array with higher axial frequency discarded
    """
    norm_factor = jnp.sqrt((array3d.shape[2] - 2 * threshold) / array3d.shape[2])
    ft_array3d = jnp.fft.fftshift(jnp.fft.fftn(array3d, norm="ortho"))
    discarded = ft_array3d[:, :, threshold:-threshold]
    del array3d, ft_array3d
    discarded *= norm_factor
    return jnp.fft.ifftn(jnp.fft.ifftshift(discarded), norm="ortho")


def zeropad_higher_axial_freq(
    array3d: Array,
    number: int,
) -> Array:
    """Zero pad the higher axial frequency.

    Parameters
    ----------
    array3d : `Array`
        3D array to be zero padded
    number : `int`
        Number of pixels to be zero padded

    Returns
    -------
    `Array`
        Zero padded 3D array
    """
    norm_factor = jnp.sqrt((array3d.shape[2] + 2 * number) / array3d.shape[2])
    ft_array3d = jnp.fft.fftshift(jnp.fft.fftn(array3d, norm="ortho"))
    del array3d
    ft_array3d = jnp.pad(
        ft_array3d,
        ((0, 0), (0, 0), (number, number)),
        mode="constant",
        constant_values=0,
    )
    ft_array3d *= norm_factor
    return jnp.fft.ifftn(jnp.fft.ifftshift(ft_array3d), norm="ortho")


def _find_max_args(array: Array) -> tuple[int, int, float]:
    max_value = float(jnp.max(array))
    max_index = jnp.unravel_index(jnp.argmax(array), array.shape)
    max_x = int(max_index[0])
    max_y = int(max_index[1])
    return max_x, max_y, max_value


def _warn_deprecated_odt_api(message: str) -> None:
    warnings.warn(message, FutureWarning, stacklevel=2)


def _shift_dh_spectrum(
    params: ODTParameters, cp_spectrum: Array, illumination_vector: tuple[int, int], edge_size: int = 0
) -> Array:
    # TODO(@fukushima-ilab): I should treat fft factor change induced by a slight shape change here (#107)
    expanded_cp_spectrum = jnp.zeros(
        (2 * params.aperturesize_px + 1 + 2 * edge_size, 2 * params.aperturesize_px + 1 + 2 * edge_size),
        dtype=cp_spectrum.dtype,
    )
    center_x = expanded_cp_spectrum.shape[0] // 2
    center_y = expanded_cp_spectrum.shape[1] // 2
    start_x = center_x - cp_spectrum.shape[0] // 2 - illumination_vector[0]
    start_y = center_y - cp_spectrum.shape[1] // 2 - illumination_vector[1]

    return expanded_cp_spectrum.at[
        start_x : start_x + cp_spectrum.shape[0],
        start_y : start_y + cp_spectrum.shape[1],
    ].set(cp_spectrum)


def _calc_1st_scattering_spectrum(
    cp_field: Array,
    ref_cp_field: Array,
    params: ODTParameters,
    approx_type: str,
    illumination_vector: tuple[int, int],
    edge_size: int = 0,
    offset_regions: OffsetRegions = None,
    gradient_correction: bool = True,
) -> Array:
    if edge_size != 0:
        cp_field = cp_field[edge_size:-edge_size, edge_size:-edge_size]
        ref_cp_field = ref_cp_field[edge_size:-edge_size, edge_size:-edge_size]

    if approx_type == "Born":
        scattering_field = (cp_field - ref_cp_field) / jnp.where(jnp.abs(ref_cp_field) < EPSILON, EPSILON, ref_cp_field)
    elif approx_type == "Rytov":
        scattering_field = _log_field(cp_field, ref_cp_field)
    else:
        msg = f"Unknown approximation type: {approx_type}"
        raise ValueError(msg)

    # offset correction
    if offset_regions:
        scattering_field = correct_offset(scattering_field, offset_regions)

    if gradient_correction:
        scattering_field = correct_gradient(scattering_field, edge_size=edge_size)

    scattering_spectrum = (
        jnp.fft.fftshift(jnp.fft.fft2(scattering_field, norm="ortho"))
        * (params.cpfield_xy2spectrum) ** 2
        * (2 * jnp.pi)
    )  # last factor is to adjust to the non-Unitary derivation in Tamamitsu's paper
    mask_for_synthesis = make_disk(
        (
            cp_field.shape[0] // 2 - illumination_vector[0],
            cp_field.shape[1] // 2 - illumination_vector[1],
        ),
        params.aperturesize_px // 2,
        cp_field.shape[0],
    )
    return scattering_spectrum * mask_for_synthesis


def _log_field(cp_field: Array, ref_cp_field: Array) -> Array:
    field_log = jnp.log(jnp.maximum(jnp.abs(cp_field), EPSILON)) + 1j * jnp.angle(cp_field)
    ref_field_log = jnp.log(jnp.maximum(jnp.abs(ref_cp_field), EPSILON)) + 1j * jnp.angle(ref_cp_field)

    amplitude = jnp.real(field_log) - jnp.real(ref_field_log)
    phase = unwrap_phase(jnp.imag(field_log) - jnp.imag(ref_field_log))

    return amplitude + 1j * phase


def _embed_3d_spectrum_with_weights(
    spectrum2d: Array,
    shape_3d: tuple[int, int, int],
    params: ODTParameters,
    illumination_vector: tuple[int, int],
    mode: str,
    ewald_embedding_mode: EwaldEmbeddingMode,
) -> tuple[Array, Array]:
    embedding_weight = _calc_ewald_embedding_weight(
        shape_3d,
        params,
        illumination_vector,
        mode,
        ewald_embedding_mode,
    )
    array_tiled = jnp.stack([spectrum2d] * shape_3d[2], axis=2)
    return array_tiled * embedding_weight, embedding_weight


def _calc_ewald_embedding_weight(
    shape_3d: tuple[int, int, int],
    params: ODTParameters,
    illumination_vector: tuple[int, int],
    mode: str,
    ewald_embedding_mode: EwaldEmbeddingMode,
) -> Array:
    xx, yy, zz = _make_ewald_coordinate_grids(shape_3d)
    circle = (xx + illumination_vector[0]) ** 2 + (yy + illumination_vector[1]) ** 2 <= (
        params.aperturesize_px // 2
    ) ** 2
    fz_circle = _calc_ewald_fz_circle(xx, yy, params, illumination_vector, mode)

    if ewald_embedding_mode == EwaldEmbeddingMode.TRUNCATE:
        return _calc_single_plane_ewald_embedding_weight(
            zz,
            circle,
            fz_circle.astype(jnp.int32),
            shape_3d[2],
        ).astype(fz_circle.dtype)

    if ewald_embedding_mode == EwaldEmbeddingMode.NEAREST:
        fz_nearest = _round_half_away_from_zero(fz_circle).astype(jnp.int32)
        return _calc_single_plane_ewald_embedding_weight(zz, circle, fz_nearest, shape_3d[2]).astype(fz_circle.dtype)

    if ewald_embedding_mode == EwaldEmbeddingMode.LINEAR:
        return _calc_linear_ewald_embedding_weight(zz, circle, fz_circle, shape_3d[2])

    msg = f"Unknown Ewald embedding mode: {ewald_embedding_mode}"
    raise ValueError(msg)


def _make_ewald_coordinate_grids(shape_3d: tuple[int, int, int]) -> tuple[Array, Array, Array]:
    xx, yy = jnp.meshgrid(
        jnp.arange(-shape_3d[0] // 2 + 1, shape_3d[0] // 2 + 1),
        jnp.arange(-shape_3d[1] // 2 + 1, shape_3d[1] // 2 + 1),
        indexing="ij",
    )

    _, _, zz = jnp.meshgrid(
        jnp.arange(-shape_3d[0] // 2 + 1, shape_3d[0] // 2 + 1),
        jnp.arange(-shape_3d[1] // 2 + 1, shape_3d[1] // 2 + 1),
        jnp.arange(-shape_3d[2] // 2 + 1, shape_3d[2] // 2 + 1),
        indexing="ij",
    )

    return xx, yy, zz


def _calc_ewald_fz_circle(
    xx: Array,
    yy: Array,
    params: ODTParameters,
    illumination_vector: tuple[int, int],
    mode: str,
) -> Array:
    fz_circle = jnp.sqrt(
        params.light_freq_px**2 - (xx + illumination_vector[0]) ** 2 - (yy + illumination_vector[1]) ** 2
    ) - jnp.sqrt(params.light_freq_px**2 - illumination_vector[0] ** 2 - illumination_vector[1] ** 2)

    if mode == "Backward":
        return -fz_circle

    return fz_circle


def _calc_single_plane_ewald_embedding_weight(
    zz: Array,
    circle: Array,
    fz_plane: Array,
    axial_size: int,
) -> Array:
    fz_tile = _tile_z_plane(fz_plane, axial_size)
    return jnp.asarray((zz == fz_tile) & circle[:, :, None], dtype=fz_tile.dtype)


def _calc_linear_ewald_embedding_weight(zz: Array, circle: Array, fz_circle: Array, axial_size: int) -> Array:
    fz_floor = jnp.floor(fz_circle)
    fz_ceil = fz_floor + 1
    lower_weight = fz_ceil - fz_circle
    upper_weight = fz_circle - fz_floor
    lower_tile = _tile_z_plane(fz_floor.astype(jnp.int32), axial_size)
    upper_tile = _tile_z_plane(fz_ceil.astype(jnp.int32), axial_size)
    lower_weight_tile = _tile_z_plane(lower_weight, axial_size)
    upper_weight_tile = _tile_z_plane(upper_weight, axial_size)
    circle_tile = jnp.asarray(circle[:, :, None], dtype=fz_circle.dtype)
    return (
        jnp.asarray(zz == lower_tile, dtype=fz_circle.dtype) * lower_weight_tile
        + jnp.asarray(zz == upper_tile, dtype=fz_circle.dtype) * upper_weight_tile
    ) * circle_tile


def _tile_z_plane(array2d: Array, axial_size: int) -> Array:
    return jnp.tile(array2d, (axial_size, 1, 1)).transpose(1, 2, 0)


def _round_half_away_from_zero(array: Array) -> Array:
    return jnp.sign(array) * jnp.floor(jnp.abs(array) + 0.5)


def _calc_kz_disk(
    params: ODTParameters,
    shape: int | tuple[int, int],
    illumination_vector: tuple[int, int],
    precision: ArrayPrecision,
) -> Array:
    if isinstance(shape, int):
        shape = (shape, shape)
    xx, yy = jnp.meshgrid(
        jnp.arange(-shape[0] // 2 + 1, shape[0] // 2 + 1),
        jnp.arange(-shape[1] // 2 + 1, shape[1] // 2 + 1),
        indexing="ij",
    )

    disk = (xx + illumination_vector[0]) ** 2 + (yy + illumination_vector[1]) ** 2
    disk_mask = disk <= (params.aperturesize_px // 2) ** 2
    fz_disk = (params.light_freq_px**2 - disk) * disk_mask
    fz_disk = jnp.where(fz_disk < 0, 0, fz_disk)
    kz_disk = fz_disk**0.5 * params.k_per_px
    return jnp.asarray(kz_disk, dtype=precision.float_precision())
