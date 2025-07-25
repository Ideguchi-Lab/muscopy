"""Optical Diffraction Tomography (ODT) calculator.

This module provides:

- `ODTParameters`: A class to store ODT parameters.
- `ODTConfig`: A class to store ODT configuration.
- `ScatteringSpectrum`: A class to store scattering spectrum.
- `synthesize_spectrum`: A function to synthesize scattering spectrums into 3D scattering potential.
- `fill_hermite_components`: A function to fill the hermite conjugated spectrum for transparent sample.
- `calc_refractive_index`: A function to calculate the refractive index from the scattering potential.
- `odt`: A function to perform ODT reconstruction.
- `calculate_odt_difference`: A function to calculate the difference between two ODT reconstructions.
- `pt_signal_1st_order`: A function to calculate the Photothermal signal with 1st order approximation.
- `discard_higher_axial_freq`: A function to discard higher axial frequency.
- `zeropad_higher_axial_freq`: A function to zero pad the higher axial frequency.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from muscopy.cfg import ArrayPrecision, OffsetRegions
from muscopy.dh import MuParameters, make_disk
from muscopy.qpi_utils import unwrap_phase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


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
        return 2 * self.aperturesize_px // 2 + 1

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
    hermite_symmetry : `bool`, optional
        Whether to use Hermite embedding or not.
    precision : `ArrayPrecision`
        Precision configuration
    edge_size : `int`, optional
        Size of removed edge in FT calculation.
    offset_regions : `OffsetRegions`, optional
        Regions to be used for offset calculation.
    """

    approx_type: str = "Born"
    hermite_symmetry: bool = True
    precision: ArrayPrecision = dataclasses.field(default_factory=ArrayPrecision)
    edge_size: int = 0
    offset_regions: OffsetRegions = None


@dataclasses.dataclass
class ScatteringSpectrum:
    r"""Data class to store scattering spectrum.

    Attributes
    ----------
    array : `Array`
        Scattering spectrum
    illumination_vector : `tuple`\[`int`, `int`\]
        Illumination vector in px unit
    """

    array: Array
    illumination_vector: tuple[int, int]


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
            2 * params.aperturesize_px + 1 - config.edge_size,
            2 * params.aperturesize_px + 1 - config.edge_size,
            params.freq_axial_extent_px,
        ),
        dtype=config.precision.complex_precision(),
    )
    synthesized_weight = jnp.ones_like(synthesized_spectrum, dtype=config.precision.int_precision())

    print("Synthesize spectrum...")  # noqa: T201
    for scattering_spectrum in tqdm(scattering_spectrums):
        kz_disk = _calc_kz_disk(
            params, scattering_spectrum.array.shape[0], scattering_spectrum.illumination_vector, config.precision
        )
        scattering_potential = _embed_3d_spectrum(
            scattering_spectrum.array * 2j * kz_disk,
            (synthesized_spectrum.shape[0], synthesized_spectrum.shape[1], synthesized_spectrum.shape[2]),
            params,
            scattering_spectrum.illumination_vector,
            mode=mode,
        )

        synthesized_spectrum = synthesized_spectrum + scattering_potential  # noqa: PLR6104
        synthesized_weight = synthesized_weight + (scattering_potential != 0)  # noqa: PLR6104

    synthesized_weight = synthesized_weight - (synthesized_weight > 1)  # noqa: PLR6104
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
        jnp.ones_like(scattering_potential) + scattering_potential / (params.light_freq_px * params.freq_per_px) ** 2
    )


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
    params.verify_parameters()
    # weak scattering approximation
    scattering_spectrums = []
    for cp_spectrum, ref_cp_spectrum in zip(cp_spectrums, ref_cp_spectrums, strict=False):
        max_x, max_y, _ = _find_max_args(jnp.abs(ref_cp_spectrum))
        illumination_vector = (max_x - params.aperturesize_px // 2, max_y - params.aperturesize_px // 2)
        expanded_cp_spectrum = _shift_dh_spectrum(params, cp_spectrum, illumination_vector).astype(
            config.precision.complex_precision()
        )
        expanded_ref_cp_spectrum = _shift_dh_spectrum(params, ref_cp_spectrum, illumination_vector).astype(
            config.precision.complex_precision()
        )
        cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(expanded_cp_spectrum), norm="ortho")
        ref_cp_field = jnp.fft.ifft2(jnp.fft.ifftshift(expanded_ref_cp_spectrum), norm="ortho")
        scattering_spectrum_array = _calc_1st_scattering_spectrum(
            cp_field, ref_cp_field, params, config.approx_type, illumination_vector, config.offset_regions
        )
        scattering_spectrum = ScatteringSpectrum(scattering_spectrum_array, illumination_vector)
        scattering_spectrums.append(scattering_spectrum)

    synthesized_spectrum = synthesize_spectrum(scattering_spectrums, params, config, mode="Forward")

    if config.hermite_symmetry:
        synthesized_spectrum = fill_hermite_components(synthesized_spectrum)

    scattering_potential = jnp.fft.ifftn(jnp.fft.ifftshift(synthesized_spectrum), norm="ortho")

    scattering_potential = jnp.fft.fftshift(scattering_potential, axes=(2))

    factor = params.spectrum2cpfield_xy**2 * params.spectrum2cpfield_z / (2 * jnp.pi) ** 3
    scattering_potential *= factor

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

    print("Reconstructing ODT from dataset 1...")  # noqa: T201
    refractive_index_1, _ = odt(cp_spectrums_1, ref_cp_spectrums_1, params, config)

    print("Reconstructing ODT from dataset 2...")  # noqa: T201
    refractive_index_2, _ = odt(cp_spectrums_2, ref_cp_spectrums_2, params, config)

    # Calculate the difference
    refractive_index_diff = refractive_index_1 - refractive_index_2

    return refractive_index_diff, refractive_index_1, refractive_index_2


def pt_signal_1st_order(
    cp_spectrums_hot: Sequence[Array],
    cp_spectrums_cold: Sequence[Array],
    ref_cp_spectrums: Sequence[Array],
    params: ODTParameters,
    config: ODTConfig,
) -> tuple[Array, Array]:
    r"""Calculate the Photothermal signal with 1st order approximation.

    Parameters
    ----------
    cp_spectrums_hot : `collections.abc.Sequence`\[`Array`\]
        Complex field spectrum of the hot sample
    cp_spectrums_cold : `collections.abc.Sequence`\[`Array`\]
        Complex field spectrum of the cold sample
    ref_cp_spectrums : `collections.abc.Sequence`\[`Array`\]
        Reference complex field spectrum
    params : `ODTParameters`
        ODT parameter instance
    config : `ODTConfig`
        ODT configuration

    Returns
    -------
    `tuple`\[`jax.Array`, `jax.Array`\]
        Photothermal signal and its Fourier transform

    Raises
    ------
    ValueError
        If the number of spectrums in hot and cold datasets do not match,
        or if the number of reference spectrums does not match the number of spectrums in hot and cold datasets.
    """
    if len(cp_spectrums_hot) != len(cp_spectrums_cold):
        msg = "The number of spectrums in hot and cold datasets must be the same"
        raise ValueError(msg)
    if len(ref_cp_spectrums) != len(cp_spectrums_hot):
        msg = "The number of reference spectrums must match the number of spectrums in hot and cold datasets"
        raise ValueError(msg)
    _, synthesized_spectrum_hot = odt(cp_spectrums_hot, ref_cp_spectrums, params, config)
    _, synthesized_spectrum_cold = odt(cp_spectrums_cold, ref_cp_spectrums, params, config)

    scattering_potential_pt = synthesized_spectrum_hot - synthesized_spectrum_cold
    ft_pt_signal = scattering_potential_pt / (params.light_freq_px * params.k_per_px) ** 2 / params.n_sol * 2 * jnp.pi
    pt_signal = jnp.fft.ifftn(jnp.fft.ifftshift(ft_pt_signal), norm="ortho")

    factor = params.spectrum2cpfield_xy**2 * params.spectrum2cpfield_z / (2 * jnp.pi) ** 3
    pt_signal *= factor

    return pt_signal, ft_pt_signal


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


def _shift_dh_spectrum(params: ODTParameters, cp_spectrum: Array, illumination_vector: tuple[int, int]) -> Array:
    expanded_cp_spectrum = jnp.zeros(
        (2 * params.aperturesize_px + 1, 2 * params.aperturesize_px + 1), dtype=cp_spectrum.dtype
    )

    return expanded_cp_spectrum.at[
        params.aperturesize_px // 2 - illumination_vector[0] : 3 * params.aperturesize_px // 2 - illumination_vector[0],
        params.aperturesize_px // 2 - illumination_vector[1] : 3 * params.aperturesize_px // 2 - illumination_vector[1],
    ].set(cp_spectrum)


def _calc_1st_scattering_spectrum(
    cp_field: Array,
    ref_cp_field: Array,
    params: ODTParameters,
    approx_type: str,
    illumination_vector: tuple[int, int],
    offset_regions: OffsetRegions = None,
) -> Array:
    if approx_type == "Born":
        scattering_field = (cp_field - ref_cp_field) / ref_cp_field
    elif approx_type == "Rytov":
        scattering_field = _log_field(cp_field, ref_cp_field)
    else:
        msg = f"Unknown approximation type: {approx_type}"
        raise ValueError(msg)

    # offset correction
    if offset_regions:
        amplitude_offset = 0.0
        for region in offset_regions:
            amplitude_offset += float(
                jnp.mean(jnp.abs(scattering_field[region[0][0] : region[0][1], region[1][0] : region[1][1]]))
            )
        amplitude_offset /= len(offset_regions)
        scattering_field = scattering_field - amplitude_offset  # noqa: PLR6104

    scattering_spectrum = (
        jnp.fft.fftshift(jnp.fft.fft2(scattering_field, norm="ortho"))
        * (params.cpfield_xy2spectrum) ** 2
        * (2 * jnp.pi)
    )  # last factor is to adjust to the non-Unitary derivation in Tamamitsu's paper
    mask_for_synthesis = make_disk(
        (
            params.aperturesize_px + illumination_vector[0],
            params.aperturesize_px + illumination_vector[1],
        ),
        params.aperturesize_px // 2,
        cp_field.shape[0],
    )
    return scattering_spectrum * mask_for_synthesis


def _log_field(cp_field: Array, ref_cp_field: Array) -> Array:
    field_log = jnp.log(cp_field + 1e-40)
    field_log_real = jnp.real(field_log)
    field_log_imag = jnp.imag(field_log)
    ref_field_log = jnp.log(ref_cp_field + 1e-40)
    ref_field_log_real = jnp.real(ref_field_log)
    ref_field_log_imag = jnp.imag(ref_field_log)

    amplitude = field_log_real - ref_field_log_real
    phase = unwrap_phase(field_log_imag - ref_field_log_imag)

    return amplitude + 1j * phase


# @jax.jit
def _embed_3d_spectrum(
    spectrum2d: Array,
    shape_3d: tuple[int, int, int],
    params: ODTParameters,
    illumination_vector: tuple[int, int],
    mode: str,
) -> Array:
    xx, yy = jnp.meshgrid(
        jnp.arange(-shape_3d[0] // 2, shape_3d[0] // 2),
        jnp.arange(
            -shape_3d[1] // 2,
            shape_3d[1] // 2,
        ),
        indexing="ij",
    )

    _, _, zz = jnp.meshgrid(
        jnp.arange(-shape_3d[0] // 2, shape_3d[0] // 2),
        jnp.arange(-shape_3d[1] // 2, shape_3d[1] // 2),
        jnp.arange(-shape_3d[2] // 2, shape_3d[2] // 2),
        indexing="ij",
    )

    circle = (xx - illumination_vector[0]) ** 2 + (yy - illumination_vector[1]) ** 2 <= (
        params.aperturesize_px // 2
    ) ** 2

    fz_circle = jnp.sqrt(
        params.light_freq_px**2 - (xx - illumination_vector[0]) ** 2 - (yy - illumination_vector[1]) ** 2
    ) - jnp.sqrt(params.light_freq_px**2 - illumination_vector[0] ** 2 - illumination_vector[1] ** 2)

    if mode == "Backward":
        fz_circle = -fz_circle

    fz_value = fz_circle * circle
    fz_value -= (1 - circle) * 2 * params.freq_axial_extent_px
    fz_tile = jnp.tile(fz_value, (shape_3d[2], 1, 1))
    fz_tile = fz_tile.transpose(1, 2, 0)
    fz_tile = fz_tile.astype(jnp.int32)  # necessary for the equivalence check

    fz_index = zz == fz_tile

    array_tiled = jnp.stack([spectrum2d] * shape_3d[2], axis=2)
    return array_tiled * fz_index


def _calc_kz_disk(
    params: ODTParameters,
    shape: int | tuple[int, int],
    illumination_vector: tuple[int, int],
    precision: ArrayPrecision,
) -> Array:
    if isinstance(shape, int):
        shape = (shape, shape)
    xx, yy = jnp.meshgrid(
        jnp.arange(-shape[0] // 2, shape[0] // 2),
        jnp.arange(-shape[1] // 2, shape[1] // 2),
        indexing="ij",
    )

    disk = (xx - illumination_vector[0]) ** 2 + (yy - illumination_vector[1]) ** 2
    disk_mask = disk <= (params.aperturesize_px // 2) ** 2
    fz_disk = (params.light_freq_px**2 - disk) * disk_mask
    fz_disk = jnp.where(fz_disk < 0, 0, fz_disk)
    kz_disk = fz_disk**0.5 * params.k_per_px
    return kz_disk.astype(precision.float_precision())
