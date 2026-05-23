"""Intensity Diffraction Tomography (IDT) module."""

from __future__ import annotations

import dataclasses
import warnings
from enum import StrEnum
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from muscopy.cfg import ArrayPrecision
from muscopy.odt import ODTParameters

if TYPE_CHECKING:
    from collections.abc import Sequence

# Numerical stability constants
_DEFAULT_NORMALIZATION_EPSILON = 1e-6  # Small positive value to avoid division by zero during normalization
_GREEN_FUNC_DENOMINATOR_EPSILON = 1e-6
_ILLUMINATION_TOL = 1e-9
_IMAGE_NDIM = 2


class IDTZCenteringMode(StrEnum):
    """Axial z-grid convention for IDT reconstruction.

    The mode controls how reconstructed slice indices are mapped to physical
    axial positions before evaluating the IDT transfer functions. For
    ``idx_z`` in ``range(num_z_slices)`` and axial spacing ``dz``:

    ``CENTRAL_SLICE_ZERO``
        Uses ``z = (idx_z - num_z_slices // 2) * dz``. This preserves the
        historical IDT behavior and guarantees that slice
        ``num_z_slices // 2`` is exactly at ``z = 0``. For an odd number of
        slices the grid is symmetric around zero. For an even number of slices
        the grid contains a zero slice but has one more negative sample than
        positive sample, so the full volume midpoint is shifted by
        ``-0.5 * dz``.

    ``SYMMETRIC_VOLUME``
        Uses ``z = (idx_z - (num_z_slices - 1) / 2) * dz``. This centers the
        reconstructed volume endpoints around ``z = 0``. For an odd number of
        slices the central slice is exactly at zero. For an even number of
        slices there is no zero slice; the two middle slices are placed at
        ``-0.5 * dz`` and ``+0.5 * dz``.
    """

    CENTRAL_SLICE_ZERO = "central_slice_zero"
    SYMMETRIC_VOLUME = "symmetric_volume"


@dataclasses.dataclass
class IDTParameters(ODTParameters):
    """IDT Parameters.

    Attributes
    ----------
    num_z_slices : int
        Number of reconstructed axial slices.
    aperturesize_px : int
        Radius of the cropped IDT intensity-spectrum support. In IDT this
        support is approximately twice the coherent pupil radius.

    Notes
    -----
    IDT frequency coordinates such as ``kx``, ``ky``, illumination shifts, and
    ``light_freq_px`` are represented in frequency-pixel units. ``k_per_px``
    converts frequency-pixel units to physical phase units. Axial positions
    and ``imgpx_axial_m_per_px`` are represented in meters.
    """

    num_z_slices: int = 10

    @property
    def intensity_support_radius_px(self) -> float:
        """Radius of the cropped IDT intensity spectrum support."""
        return idt_intensity_support_radius_px(self)

    @property
    def coherent_pupil_radius_px(self) -> float:
        """Radius of the coherent pupil support."""
        return idt_coherent_pupil_radius_px(self)


@dataclasses.dataclass
class IDTConfig:
    """Intensity Diffraction Tomography (IDT) configuration.

    Attributes
    ----------
    precision : `ArrayPrecision`
        Precision configuration. Defaults to 32-bit float and complex arrays.
    ref_floor_ratio : float
        Minimum valid reference intensity as a fraction of the mean absolute
        reference intensity.
    g_clip : float | None
        Optional symmetric clipping threshold for normalized intensity contrast.
        Defaults to ``None`` so normalized contrast is not clipped unless
        explicitly requested.
    normalization_epsilon : float
        Small positive floor used during normalization to avoid division by zero.
    determinant_rel_floor : float
        Relative floor for the coupled inverse determinant.
    z_centering : IDTZCenteringMode | str
        Axial z-grid convention used when mapping reconstruction slice
        indices to physical z positions. ``"central_slice_zero"`` preserves
        the historical convention where index ``num_z_slices // 2`` is
        exactly ``z = 0``. ``"symmetric_volume"`` centers the full stack around
        ``z = 0``; with an even number of slices, zero lies halfway between the
        two central slices.
    check_ifft_imag_residual : bool
        Whether to warn when the spatial-domain inverse FFT has a large
        imaginary residual before taking the real part.
    imag_residual_warn_threshold : float
        Relative imaginary residual threshold for the optional diagnostic.
    """

    precision: ArrayPrecision = dataclasses.field(default_factory=ArrayPrecision)
    ref_floor_ratio: float = 0.0
    g_clip: float | None = None
    normalization_epsilon: float = _DEFAULT_NORMALIZATION_EPSILON
    determinant_rel_floor: float = 1e-4
    z_centering: IDTZCenteringMode | str = IDTZCenteringMode.CENTRAL_SLICE_ZERO
    check_ifft_imag_residual: bool = False
    imag_residual_warn_threshold: float = 1e-4

    def __post_init__(self) -> None:
        """Validate configuration fields.

        Raises
        ------
        ValueError
            If a configuration value is outside its valid range.
        """
        self.z_centering = IDTZCenteringMode(self.z_centering)
        if self.ref_floor_ratio < 0:
            msg = "ref_floor_ratio must be non-negative."
            raise ValueError(msg)
        if self.g_clip is not None and self.g_clip <= 0:
            msg = "g_clip must be positive when provided."
            raise ValueError(msg)
        if self.normalization_epsilon <= 0:
            msg = "normalization_epsilon must be positive."
            raise ValueError(msg)
        if self.determinant_rel_floor < 0:
            msg = "determinant_rel_floor must be non-negative."
            raise ValueError(msg)
        if self.imag_residual_warn_threshold <= 0:
            msg = "imag_residual_warn_threshold must be positive."
            raise ValueError(msg)


def idt_intensity_support_radius_px(params: IDTParameters) -> float:
    """Radius of the cropped IDT intensity spectrum support.

    Returns
    -------
    float
        The IDT intensity spectrum support radius in frequency-pixel units.
    """
    return params.aperturesize_px


def idt_coherent_pupil_radius_px(params: IDTParameters) -> float:
    """Radius of the coherent pupil support.

    In IDT, the intensity spectrum support is approximately twice the
    coherent pupil radius.

    Returns
    -------
    float
        The coherent pupil support radius in frequency-pixel units.
    """
    return params.aperturesize_px / 2


def validate_idt_sampling(params: IDTParameters) -> None:
    """Validate IDT sampling quantities used in transfer-function scaling.

    Raises
    ------
    ValueError
        If a required sampling quantity is not positive.
    """
    if params.imgpx_axial_m_per_px <= 0:
        msg = "imgpx_axial_m_per_px must be positive."
        raise ValueError(msg)
    if params.freq_per_px <= 0:
        msg = "freq_per_px must be positive."
        raise ValueError(msg)
    if params.k_per_px <= 0:
        msg = "k_per_px must be positive."
        raise ValueError(msg)
    if params.light_freq_px <= 0:
        msg = "light_freq_px must be positive."
        raise ValueError(msg)


def validate_idt_params(params: IDTParameters) -> None:
    """Validate IDT parameter ranges and Fourier cropping support.

    Raises
    ------
    ValueError
        If IDT parameters are invalid or Fourier cropping would exceed the
        image bounds.
    """
    if params.img_size_px <= 0:
        msg = "img_size_px must be positive."
        raise ValueError(msg)
    if params.num_z_slices <= 0:
        msg = "num_z_slices must be positive."
        raise ValueError(msg)
    if params.aperturesize_px <= 0:
        msg = "aperturesize_px must be positive."
        raise ValueError(msg)
    if 2 * params.intensity_support_radius_px + 1 > params.img_size_px:
        msg = "2 * aperturesize_px + 1 must be <= img_size_px for Fourier cropping."
        raise ValueError(msg)
    if params.n_sol <= 0:
        msg = "n_sol must be positive."
        raise ValueError(msg)
    validate_idt_sampling(params)


def make_z_position(params: IDTParameters, idx_z: int, config: IDTConfig) -> float:
    """Return the physical z position for a reconstructed slice index.

    The position is determined by ``config.z_centering``:

    - ``IDTZCenteringMode.CENTRAL_SLICE_ZERO`` maps slice ``num_z_slices // 2``
      to ``z = 0`` using ``(idx_z - num_z_slices // 2) * dz``. This is useful
      for reproducing previous reconstructions or when downstream code expects
      a slice exactly at the focal plane even for an even number of slices.
    - ``IDTZCenteringMode.SYMMETRIC_VOLUME`` maps the index range with
      ``(idx_z - (num_z_slices - 1) / 2) * dz``. This is useful when the
      reconstructed volume should be geometrically centered around the focal
      plane; for even slice counts, the focal plane lies between the two middle
      slices.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters containing ``num_z_slices`` and
        ``imgpx_axial_m_per_px``.
    idx_z : int
        Reconstructed slice index.
    config : IDTConfig
        IDT configuration containing the z-centering mode.

    Returns
    -------
    float
        The z position in meters.

    Raises
    ------
    ValueError
        If the z-centering mode is unknown.
    """
    if config.z_centering == IDTZCenteringMode.CENTRAL_SLICE_ZERO:
        z_index = float(idx_z - params.num_z_slices // 2)
    elif config.z_centering == IDTZCenteringMode.SYMMETRIC_VOLUME:
        z_index = idx_z - (params.num_z_slices - 1) / 2
    else:  # pragma: no cover - IDTConfig validation should prevent this.
        msg = f"Unknown z_centering: {config.z_centering}"
        raise ValueError(msg)
    return z_index * params.imgpx_axial_m_per_px


def validate_xy_image(params: IDTParameters, image_xy: Array, *, name: str) -> None:
    """Validate that an IDT image is a 2D internal [x, y] array.

    Raises
    ------
    ValueError
        If the image is not a 2D array with shape ``(img_size_px, img_size_px)``.
    """
    if image_xy.ndim != _IMAGE_NDIM:
        msg = f"{name} must be a 2D [x, y] array."
        raise ValueError(msg)
    expected_shape = (params.img_size_px, params.img_size_px)
    if image_xy.shape != expected_shape:
        msg = f"{name} must have shape {expected_shape} in [x, y] order, got {image_xy.shape}."
        raise ValueError(msg)


def relative_imag_residual(arr: Array, *, normalization_epsilon: float = _DEFAULT_NORMALIZATION_EPSILON) -> Array:
    """Return the relative imaginary residual of a complex array.

    Parameters
    ----------
    arr : Array
        Complex array to evaluate.
    normalization_epsilon : float, optional
        Small positive floor for the real norm denominator.

    Returns
    -------
    Array
        ``||imag(arr)|| / max(||real(arr)||, epsilon)``.

    Raises
    ------
    ValueError
        If ``normalization_epsilon`` is not positive.
    """
    if normalization_epsilon <= 0:
        msg = "normalization_epsilon must be positive."
        raise ValueError(msg)
    real_norm = jnp.linalg.norm(jnp.real(arr))
    imag_norm = jnp.linalg.norm(jnp.imag(arr))
    return jnp.asarray(imag_norm / jnp.maximum(real_norm, normalization_epsilon))


def _warn_if_imag_residual_large(arr: Array, *, name: str, config: IDTConfig) -> None:
    if not config.check_ifft_imag_residual:
        return
    residual = float(relative_imag_residual(arr, normalization_epsilon=config.normalization_epsilon))
    if residual > config.imag_residual_warn_threshold:
        warnings.warn(
            f"{name} inverse FFT imaginary residual is {residual:.3g}, "
            f"above threshold {config.imag_residual_warn_threshold:.3g}.",
            RuntimeWarning,
            stacklevel=2,
        )


def make_frequency_grid_xy(
    params: IDTParameters,
    *,
    precision: ArrayPrecision,
) -> tuple[Array, Array]:
    """Return kx, ky frequency grids with shape [kx, ky].

    The grids use frequency-pixel units and span the cropped IDT intensity
    spectrum support ``[-aperturesize_px, +aperturesize_px]``.

    Returns
    -------
    tuple[Array, Array]
        The ``kx`` and ``ky`` grids in internal [x, y] order.
    """
    validate_idt_params(params)
    precision.validate()
    dtype = precision.float_precision()
    support_radius_px = params.intensity_support_radius_px
    coords_x = jnp.arange(-support_radius_px, support_radius_px + 1, dtype=dtype)
    coords_y = jnp.arange(-support_radius_px, support_radius_px + 1, dtype=dtype)
    kx, ky = jnp.meshgrid(coords_x, coords_y, indexing="ij")
    return kx, ky


def compute_g_list(
    i_list: Sequence[Array],
    i_reference: Sequence[Array],
    normalize: bool = True,
    *,
    precision: ArrayPrecision | None = None,
    ref_floor_ratio: float = 0.0,
    g_clip: float | None = None,
    normalization_epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> list[Array]:
    """Compute list of intensity constrasts g_l for each illumination angle.

    Parameters
    ----------
    i_list : list of [x, y] arrays of Intensity images
        under different angles
    i_reference : list of [x, y] background intensity images for subtraction
    normalize : bool
        whether or not to normalize
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned arrays.
    ref_floor_ratio : float, optional
        Minimum valid reference intensity as a fraction of the mean absolute
        reference intensity.
    g_clip : float | None, optional
        Optional symmetric clipping threshold for normalized contrast. Defaults
        to ``None`` so normalized contrast is not clipped unless explicitly
        requested.
    normalization_epsilon : float, optional
        Small positive floor used during normalization to avoid division by zero.

    Returns
    -------
    g_list : list of [x, y] arrays
        Intensity different or contrast for each angle.

    Raises
    ------
    ValueError
        If normalization floor or clipping parameters are invalid, or if the
        image and reference sequences have different lengths.
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    if ref_floor_ratio < 0:
        msg = "ref_floor_ratio must be non-negative."
        raise ValueError(msg)
    if g_clip is not None and g_clip <= 0:
        msg = "g_clip must be positive when provided."
        raise ValueError(msg)
    if normalization_epsilon <= 0:
        msg = "normalization_epsilon must be positive."
        raise ValueError(msg)
    if len(i_list) != len(i_reference):
        msg = "i_list and i_reference must have the same length."
        raise ValueError(msg)
    float_dtype = precision.float_precision()
    g_list = []
    for i_m, i_ref in zip(i_list, i_reference, strict=True):
        target_image = jnp.asarray(i_m, dtype=float_dtype)
        reference_image = jnp.asarray(i_ref, dtype=float_dtype)
        if normalize:
            ref_scale = jnp.mean(jnp.abs(reference_image))
            ref_floor = jnp.maximum(normalization_epsilon, ref_floor_ratio * ref_scale)
            valid_ref = jnp.isfinite(target_image) & jnp.isfinite(reference_image) & (reference_image > ref_floor)
            denom = jnp.where(valid_ref, reference_image, 1.0)
            g = jnp.where(valid_ref, (target_image - reference_image) / denom, 0.0)
            if g_clip is not None:
                g = jnp.clip(g, -g_clip, g_clip)
            g_list.append(jnp.asarray(g, dtype=float_dtype))
        else:
            g = target_image - reference_image
            g_list.append(jnp.asarray(g, dtype=float_dtype))
    return g_list


def fourier_transform(
    params: IDTParameters,
    g_list: Sequence[Array],
    *,
    precision: ArrayPrecision | None = None,
) -> list[Array]:
    """
    Compute the Fourier Transform of each g_l in g_list.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    g_list : list of [x, y] arrays
        Intensity differences per angle in the internal spatial domain.
    precision : `ArrayPrecision`, optional
        Precision configuration for the Fourier arrays.

    Returns
    -------
    g_tilde_list : list of [kx, ky] arrays
        Fourier Transforms of g_l in the internal frequency domain.
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    float_dtype = precision.float_precision()
    complex_dtype = precision.complex_precision()
    g_tilde_list = []
    support_radius_px = params.intensity_support_radius_px
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    incoherent_limit_mask = kx**2 + ky**2 <= support_radius_px**2
    ft_scaling_factor = jnp.asarray(jnp.sqrt((2 * support_radius_px + 1) / params.img_size_px), dtype=float_dtype)
    for g_l in g_list:
        g_array = jnp.asarray(g_l, dtype=float_dtype)
        g_tilde = jnp.fft.fftshift(jnp.fft.fft2(g_array, norm="ortho"))  # FFT with fftshift for centered spectrum
        g_tilde_cropped = g_tilde[
            params.img_size_px // 2 - params.aperturesize_px : params.img_size_px // 2 + params.aperturesize_px + 1,
            params.img_size_px // 2 - params.aperturesize_px : params.img_size_px // 2 + params.aperturesize_px + 1,
        ]
        g_tilde_masked = g_tilde_cropped * incoherent_limit_mask * ft_scaling_factor**2
        g_tilde_list.append(jnp.asarray(g_tilde_masked, dtype=complex_dtype))
    return g_tilde_list


def make_green_func(
    params: IDTParameters,
    u_shift: tuple[float, float],
    z: float,
    *,
    precision: ArrayPrecision | None = None,
) -> Array:
    """Generate the Green's function for a given illumination angle.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    u_shift : tuple[float, float]
        The illumination shift in frequency-pixel units as (u_x, u_y).
    z : float
        The axial position in meters.
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned array.

    Returns
    -------
    Array
        The Green's function evaluated at the given illumination angle.
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    complex_dtype = precision.complex_precision()
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    ux = kx + u_shift[0]
    uy = ky + u_shift[1]
    uz_squared = params.light_freq_px**2 - ux**2 - uy**2

    pupil_radius_px = params.coherent_pupil_radius_px
    in_shifted_pupil = ux**2 + uy**2 <= pupil_radius_px**2
    propagating = uz_squared > 0
    valid = in_shifted_pupil & propagating

    uz = jnp.sqrt(jnp.maximum(uz_squared, 0.0))
    green_scale = 1.0 / (4.0 * jnp.pi * params.freq_per_px)
    green_func = jnp.where(
        valid,
        green_scale * jnp.exp(-1j * uz * z * params.k_per_px) / jnp.maximum(uz, _GREEN_FUNC_DENOMINATOR_EPSILON),
        0.0 + 0.0j,
    )
    return jnp.asarray(green_func, dtype=complex_dtype)


def make_pupil_func(
    params: IDTParameters,
    u_shift: tuple[float, float],
    *,
    precision: ArrayPrecision | None = None,
) -> Array:
    """Create pupil function for IDT reconstruction.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters containing aperture size information
    u_shift : tuple[float, float]
        Shift in frequency-pixel units as (u_x, u_y).
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned array.

    Returns
    -------
    Array
        Pupil function with shape [x, y].
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    kx, ky = make_frequency_grid_xy(params, precision=precision)
    ux = kx + u_shift[0]
    uy = ky + u_shift[1]
    pupil_radius_px = params.coherent_pupil_radius_px
    pupil = ux**2 + uy**2 <= pupil_radius_px**2
    return jnp.asarray(pupil, dtype=precision.float_precision())


def transfer_func_re(
    params: IDTParameters,
    u_illumination: tuple[float, float],
    z: float,
    incident_intensity: float,
    *,
    precision: ArrayPrecision | None = None,
) -> Array:
    """Compute real part of transfer function for IDT.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters
    u_illumination : tuple[float, float]
        Illumination angle in frequency-pixel units as (u_x, u_y).
    z : float
        Depth position in meters.
    incident_intensity : float
        Incident intensity
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned array.

    Returns
    -------
    Array
        Real part of transfer function

    Raises
    ------
    ValueError
        If illumination angle results in invalid z-component
    """
    u_ill_x, u_ill_y = u_illumination
    u_ill_z_squared = params.light_freq_px**2 - u_ill_x**2 - u_ill_y**2
    if u_ill_z_squared < -_ILLUMINATION_TOL:
        msg = f"Invalid illumination angle {u_illumination}: u_ill_z_squared must be non-negative."
        raise ValueError(msg)
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    u_ill_z = jnp.sqrt(jnp.maximum(u_ill_z_squared, 0.0))
    first_term = (
        make_green_func(params, (-u_ill_x, -u_ill_y), z, precision=precision)
        * jnp.exp(1j * u_ill_z * z * params.k_per_px)
        * make_pupil_func(params, (-u_ill_x, -u_ill_y), precision=precision)
    )
    second_term = (
        jnp.conjugate(make_green_func(params, (u_ill_x, u_ill_y), z, precision=precision))
        * jnp.exp(-1j * u_ill_z * z * params.k_per_px)
        * jnp.conjugate(make_pupil_func(params, (u_ill_x, u_ill_y), precision=precision))
    )

    slice_thickness_m = params.imgpx_axial_m_per_px
    transfer_func = (
        1j
        * (params.k_per_px * params.light_freq_px) ** 2
        / 2
        * slice_thickness_m
        * incident_intensity
        * (first_term - second_term)
    )
    return jnp.asarray(transfer_func, dtype=precision.complex_precision())


def transfer_func_im(
    params: IDTParameters,
    u_illumination: tuple[float, float],
    z: float,
    incident_intensity: float,
    *,
    precision: ArrayPrecision | None = None,
) -> Array:
    """Compute imaginary part of transfer function for IDT.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters
    u_illumination : tuple[float, float]
        Illumination angle in frequency-pixel units as (u_x, u_y).
    z : float
        Depth position in meters.
    incident_intensity : float
        Incident intensity
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned array.

    Returns
    -------
    Array
        Imaginary part of transfer function

    Raises
    ------
    ValueError
        If illumination angle results in invalid z-component.
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    u_ill_x, u_ill_y = u_illumination
    u_ill_z_squared = params.light_freq_px**2 - u_ill_x**2 - u_ill_y**2
    if u_ill_z_squared < -_ILLUMINATION_TOL:
        msg = f"Invalid illumination angle {u_illumination}: u_ill_z_squared must be non-negative."
        raise ValueError(msg)
    u_ill_z = jnp.sqrt(jnp.maximum(u_ill_z_squared, 0.0))
    first_term = (
        make_green_func(params, (-u_ill_x, -u_ill_y), z, precision=precision)
        * jnp.exp(1j * u_ill_z * z * params.k_per_px)
        * make_pupil_func(params, (-u_ill_x, -u_ill_y), precision=precision)
    )
    second_term = (
        jnp.conjugate(make_green_func(params, (u_ill_x, u_ill_y), z, precision=precision))
        * jnp.exp(-1j * u_ill_z * z * params.k_per_px)
        * jnp.conjugate(make_pupil_func(params, (u_ill_x, u_ill_y), precision=precision))
    )

    slice_thickness_m = params.imgpx_axial_m_per_px
    transfer_func = (
        -((params.light_freq_px * params.k_per_px) ** 2)
        / 2
        * slice_thickness_m
        * incident_intensity
        * (first_term + second_term)
    )
    return jnp.asarray(transfer_func, dtype=precision.complex_precision())


def _validate_led_illumination_intensities(led_illumination_intensities: Sequence[float]) -> None:
    for idx_intensity, intensity in enumerate(led_illumination_intensities):
        if not bool(jnp.isfinite(jnp.asarray(intensity))) or intensity <= 0:
            msg = f"led_illumination_intensities[{idx_intensity}] must be positive and finite."
            raise ValueError(msg)


def compute_permittivity(  # noqa: PLR0914
    params: IDTParameters,
    g_tilde_list: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
    led_illumination_intensities: Sequence[float],
    z: float = 0.0,
    alpha: float = 1e-6,
    beta: float = 1e-6,
    *,
    precision: ArrayPrecision | None = None,
    determinant_rel_floor: float = 1e-4,
    normalization_epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[Array, Array]:
    """Compute the permittivity changes Δε_Re and Δε_Im for each slice.

    The real and imaginary permittivity components are estimated jointly by
    solving the regularized 2x2 normal equation at each spatial frequency. The
    off-diagonal coupling term between ``H_Re`` and ``H_Im`` is retained.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    g_tilde_list : list of [kx, ky] arrays
        Fourier Transforms of normalized intensity contrast per angle.
    u_illumination_list : list of tuples
        The illumination angles for each image in frequency-pixel units.
    led_illumination_intensities : list of float
        The incident intensities for each illumination angle.
    z : float, optional
        The axial position in meters, by default 0.0
    alpha : float, optional
        Regularization parameter for the real part of the permittivity, by default 1e-6
    beta : float, optional
        Regularization parameter for the imaginary part of the permittivity, by default 1e-6
    precision : `ArrayPrecision`, optional
        Precision configuration for the returned arrays.
    determinant_rel_floor : float, optional
        Relative floor for the coupled inverse determinant.
    normalization_epsilon : float, optional
        Small positive floor used during transfer-function normalization and
        inverse stability checks.

    Returns
    -------
    tuple[Array, Array]
        The computed permittivity changes Δε_Re and Δε_Im for each slice.

    Raises
    ------
    ValueError
        If the image, illumination, or intensity sequences have incompatible
        lengths, invalid determinant floor, or non-positive LED intensity.
    """
    if precision is None:
        precision = ArrayPrecision()
    precision.validate()
    if len(g_tilde_list) != len(u_illumination_list):
        msg = "g_tilde_list and u_illumination_list must have the same length."
        raise ValueError(msg)
    if len(u_illumination_list) != len(led_illumination_intensities):
        msg = "u_illumination_list and led_illumination_intensities must have the same length."
        raise ValueError(msg)
    if determinant_rel_floor < 0:
        msg = "determinant_rel_floor must be non-negative."
        raise ValueError(msg)
    if normalization_epsilon <= 0:
        msg = "normalization_epsilon must be positive."
        raise ValueError(msg)
    _validate_led_illumination_intensities(led_illumination_intensities)
    complex_dtype = precision.complex_precision()
    h_normalized_re = jnp.stack(
        [
            transfer_func_re(params, u_illumination_list[i], z, led_illumination_intensities[i], precision=precision)
            / led_illumination_intensities[i]
            for i in range(len(u_illumination_list))
        ],
        axis=-1,
    )
    h_normalized_im = jnp.stack(
        [
            transfer_func_im(params, u_illumination_list[i], z, led_illumination_intensities[i], precision=precision)
            / led_illumination_intensities[i]
            for i in range(len(u_illumination_list))
        ],
        axis=-1,
    )

    g_tilde = jnp.asarray(jnp.stack(g_tilde_list, axis=-1), dtype=complex_dtype)

    # Normalize transfer functions for the linear solve, then restore the
    # solution scale below.
    h_norm_scale = jnp.maximum(jnp.max(jnp.abs(h_normalized_re)), jnp.max(jnp.abs(h_normalized_im)))
    h_norm_scale = jnp.where(h_norm_scale < normalization_epsilon, 1.0, h_norm_scale)

    h_normalized_re_scaled = h_normalized_re / h_norm_scale
    h_normalized_im_scaled = h_normalized_im / h_norm_scale

    sum_h_re_scaled = jnp.sum(jnp.abs(h_normalized_re_scaled) ** 2, axis=-1)
    sum_h_im_scaled = jnp.sum(jnp.abs(h_normalized_im_scaled) ** 2, axis=-1)

    # Scale regularization parameters accordingly
    alpha_scaled = alpha / (h_norm_scale**2)
    beta_scaled = beta / (h_norm_scale**2)

    term1 = (sum_h_re_scaled + alpha_scaled) * (sum_h_im_scaled + beta_scaled)
    term2 = jnp.abs(jnp.sum(jnp.conjugate(h_normalized_re_scaled) * h_normalized_im_scaled, axis=-1)) ** 2
    scale_factor = jnp.maximum(jnp.real(term1 - term2), 0.0)
    scale_factor_floor = normalization_epsilon + determinant_rel_floor * jnp.real(term1)
    scale_factor_safe = jnp.maximum(scale_factor, scale_factor_floor)
    valid_support = (sum_h_re_scaled + sum_h_im_scaled) > normalization_epsilon

    # Scale the epsilon calculations accordingly
    eps_re_first_term_scaled = (sum_h_im_scaled + beta_scaled) * jnp.sum(
        jnp.conjugate(h_normalized_re_scaled) * g_tilde, axis=-1
    )
    eps_re_second_term_scaled = jnp.sum(
        jnp.conjugate(h_normalized_re_scaled) * h_normalized_im_scaled, axis=-1
    ) * jnp.sum(jnp.conjugate(h_normalized_im_scaled) * g_tilde, axis=-1)
    eps_im_first_term_scaled = (sum_h_re_scaled + alpha_scaled) * jnp.sum(
        jnp.conjugate(h_normalized_im_scaled) * g_tilde, axis=-1
    )
    eps_im_second_term_scaled = jnp.sum(
        jnp.conjugate(h_normalized_im_scaled) * h_normalized_re_scaled, axis=-1
    ) * jnp.sum(jnp.conjugate(h_normalized_re_scaled) * g_tilde, axis=-1)

    eps_re_scaled = jnp.where(
        valid_support,
        (eps_re_first_term_scaled - eps_re_second_term_scaled) / scale_factor_safe,
        0.0 + 0.0j,
    )
    eps_im_scaled = jnp.where(
        valid_support,
        (eps_im_first_term_scaled - eps_im_second_term_scaled) / scale_factor_safe,
        0.0 + 0.0j,
    )

    eps_re = eps_re_scaled / h_norm_scale
    eps_im = eps_im_scaled / h_norm_scale

    return jnp.asarray(eps_re, dtype=complex_dtype), jnp.asarray(eps_im, dtype=complex_dtype)


def compute_permitivity(
    params: IDTParameters,
    g_tilde_list: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
    led_illumination_intensities: Sequence[float],
    z: float = 0.0,
    alpha: float = 1e-6,
    beta: float = 1e-6,
    *,
    precision: ArrayPrecision | None = None,
    determinant_rel_floor: float = 1e-4,
    normalization_epsilon: float = _DEFAULT_NORMALIZATION_EPSILON,
) -> tuple[Array, Array]:
    """Backward-compatible alias for :func:`compute_permittivity`.

    Returns
    -------
    tuple[Array, Array]
        The computed permittivity changes Δε_Re and Δε_Im for each slice.
    """
    return compute_permittivity(
        params,
        g_tilde_list,
        u_illumination_list,
        led_illumination_intensities,
        z=z,
        alpha=alpha,
        beta=beta,
        precision=precision,
        determinant_rel_floor=determinant_rel_floor,
        normalization_epsilon=normalization_epsilon,
    )


def convert_to_refractive_index(eps_re: Array, eps_im: Array, n_sol: float) -> tuple[Array, Array]:
    """Convert the difference in permittivity to difference in refractive index.

    Parameters
    ----------
    eps_re : Array
        Real part of permittivity values
    eps_im : Array
        Imaginary part of permittivity values
    n_sol : float
        Refractive index of the solution

    Returns
    -------
    tuple[Array, Array]
        Real and Imaginary components of the refractive index difference

    """
    eps_complex = eps_re + 1j * eps_im
    n_complex = jnp.sqrt(n_sol**2 + eps_complex)
    n_re = jnp.real(n_complex) - n_sol
    n_im = jnp.imag(n_complex)
    return n_re, n_im


##################################################
# Main IDT computation function
##################################################


def compute_idt(  # noqa: PLR0914
    params: IDTParameters,
    intensity_images: Sequence[Array],
    ref_intensity_images: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
    *,
    config: IDTConfig | None = None,
    alpha: float = 1e-2,
    beta: float = 1e-2,
) -> tuple[Array, Array]:
    r"""Compute the refractive index from intensity images using IDT.

    Processing steps
    ----------------
    1. Collect internal [x, y] intensity images and illumination angles.
    2. Subtract and normalize the background intensity.
    3. Fourier transform each normalized intensity image.
    4. Build transfer functions for each slice depth and illumination angle.
    5. Solve the inverse problem slice by slice. The real and imaginary
       permittivity components are estimated jointly by solving the
       regularized 2x2 normal equation at each spatial frequency:

       .. math::

          \begin{bmatrix}
          H_\mathrm{Re}^* H_\mathrm{Re} + \alpha & H_\mathrm{Re}^* H_\mathrm{Im} \\
          H_\mathrm{Im}^* H_\mathrm{Re} & H_\mathrm{Im}^* H_\mathrm{Im} + \beta
          \end{bmatrix}
          \begin{bmatrix}
          \Delta \epsilon_\mathrm{Re} \\
          \Delta \epsilon_\mathrm{Im}
          \end{bmatrix}
          =
          \begin{bmatrix}
          H_\mathrm{Re}^* \tilde{g} \\
          H_\mathrm{Im}^* \tilde{g}
          \end{bmatrix}

       The off-diagonal coupling term between :math:`H_\mathrm{Re}` and
       :math:`H_\mathrm{Im}` is retained.

    6. Convert permittivity to refractive index.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    intensity_images : Sequence[Array]
        The internal [x, y] intensity images to process.
    ref_intensity_images : Sequence[Array]
        The internal [x, y] reference intensity images for comparison.
    u_illumination_list : Sequence[tuple[float, float]]
        The illumination angles for each LED as (u_x, u_y).
    config : `IDTConfig`, optional
        IDT configuration. Defaults to 32-bit precision.

    Returns
    -------
    tuple[Array, Array]
        Real and imaginary parts of the computed refractive index with shape [x, y, z].

    Raises
    ------
    ValueError
        If the input arrays have incompatible shapes.
    ValueError
        If the illumination angles are not valid.
    """
    if len(intensity_images) != len(ref_intensity_images):
        msg = "Intensity images and reference intensity images must have the same length."
        raise ValueError(msg)
    if len(intensity_images) != len(u_illumination_list):
        msg = "Intensity images and illumination angles must have the same length."
        raise ValueError(msg)
    if config is None:
        config = IDTConfig()
    validate_idt_params(params)
    config.precision.validate()
    float_dtype = config.precision.float_precision()
    complex_dtype = config.precision.complex_precision()

    # STEP 1: intensity images (input)
    for idx_image, image_xy in enumerate(intensity_images):
        validate_xy_image(params, image_xy, name=f"intensity_images[{idx_image}]")
    for idx_image, image_xy in enumerate(ref_intensity_images):
        validate_xy_image(params, image_xy, name=f"ref_intensity_images[{idx_image}]")

    # STEP 2: compute g_l

    g_l = compute_g_list(
        intensity_images,
        ref_intensity_images,
        normalize=True,
        precision=config.precision,
        ref_floor_ratio=config.ref_floor_ratio,
        g_clip=config.g_clip,
        normalization_epsilon=config.normalization_epsilon,
    )

    # STEP 3: Fourier Transform each image

    g_tilde_list = fourier_transform(params, g_l, precision=config.precision)

    # STEP 4: Build Transfer Functions

    # done in the separate functions

    # STEP 5: Solve inverse problem

    aperture_size = 2 * params.aperturesize_px + 1
    eps_re_xyz = jnp.zeros((aperture_size, aperture_size, params.num_z_slices), dtype=complex_dtype)
    eps_im_xyz = jnp.zeros((aperture_size, aperture_size, params.num_z_slices), dtype=complex_dtype)

    led_illumination_intensities = [
        float(jnp.mean(ref_image)) for ref_image in ref_intensity_images
    ]  # Assuming uniform intensity for simplicity
    _validate_led_illumination_intensities(led_illumination_intensities)

    for idx_z in range(params.num_z_slices):
        z = make_z_position(params, idx_z, config)
        eps_re, eps_im = compute_permittivity(
            params,
            g_tilde_list,
            u_illumination_list,
            led_illumination_intensities,
            z=z,
            alpha=alpha,
            beta=beta,
            precision=config.precision,
            determinant_rel_floor=config.determinant_rel_floor,
            normalization_epsilon=config.normalization_epsilon,
        )
        eps_re_xyz = eps_re_xyz.at[:, :, idx_z].set(eps_re)
        eps_im_xyz = eps_im_xyz.at[:, :, idx_z].set(eps_im)

    eps_re_xyz_spatial_complex = jnp.fft.ifft2(
        jnp.fft.ifftshift(eps_re_xyz, axes=(0, 1)),
        axes=(0, 1),
        norm="ortho",
    )
    eps_im_xyz_spatial_complex = jnp.fft.ifft2(
        jnp.fft.ifftshift(eps_im_xyz, axes=(0, 1)),
        axes=(0, 1),
        norm="ortho",
    )
    _warn_if_imag_residual_large(eps_re_xyz_spatial_complex, name="eps_re", config=config)
    _warn_if_imag_residual_large(eps_im_xyz_spatial_complex, name="eps_im", config=config)
    eps_re_xyz_spatial = jnp.real(eps_re_xyz_spatial_complex)
    eps_im_xyz_spatial = jnp.real(eps_im_xyz_spatial_complex)

    # STEP 6: Convert permittivity to refractive index

    n_re_xyz, n_im_xyz = convert_to_refractive_index(eps_re_xyz_spatial, eps_im_xyz_spatial, params.n_sol)

    # Take real part since result should be real in spatial domain
    n_re_xyz_spatial = jnp.asarray(jnp.real(n_re_xyz), dtype=float_dtype)
    n_im_xyz_spatial = jnp.asarray(jnp.real(n_im_xyz), dtype=float_dtype)

    return n_re_xyz_spatial, n_im_xyz_spatial
