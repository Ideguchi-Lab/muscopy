"""Intensity Diffraction Tomography (IDT) module."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from muscopy.dh import make_disk
from muscopy.odt import ODTParameters

if TYPE_CHECKING:
    from collections.abc import Sequence

# Numerical stability constants
_EPSILON = 1e-12  # Small value to avoid division by zero


@dataclasses.dataclass
class IDTParameters(ODTParameters):
    """IDT Parameters."""

    num_z_slices: int = 10


def compute_g_list(i_list: Sequence[Array], i_reference: Sequence[Array], normalize: bool = True) -> list[Array]:
    """Compute list of intensity constrasts g_l for each illumination angle.

    Parameters
    ----------
    i_list : list of [Nx, Ny] arrays of Intensity images
        under different angles
    i_reference : background intensity image for subtraction
    normalize : bool
        whether or not to normalize

    Returns
    -------
    g_list : list of [Nx, Ny] arrays
        Intensity different or contrast for each angle.
    """
    g_list = []
    for i_m, i_ref in zip(i_list, i_reference, strict=False):
        if normalize:
            # Avoid division by zero in normalization
            i_ref_safe = jnp.where(jnp.abs(i_ref) < _EPSILON, _EPSILON, i_ref)
            g = (i_m - i_ref) / i_ref_safe

            g_list.append(g)
    return g_list


def fourier_transform(g_list: Sequence[Array]) -> list[Array]:
    """
    Compute the Fourier Transform of each g_l in g_list.

    Parameters
    ----------
    g_list : list of [Nx, Ny] arrays
        Intensity differences per angle (spatial domain)

    Returns
    -------
    g_tilde_list : list of [Nx, Ny] arrays
        Fourier Transforms of g_l (frequency domain)
    """
    g_tilde_list = []
    for g_l in g_list:
        g_tilde = jnp.fft.fftshift(jnp.fft.fft2(g_l, norm="ortho"))  # FFT with fftshift for centered spectrum
        g_tilde_list.append(g_tilde)
    return g_tilde_list


def make_green_func(params: IDTParameters, u_shift: tuple[float, float], z: float) -> Array:
    """Generate the Green's function for a given illumination angle.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    u_shift : tuple[float, float]
        The illumination angle in the x and y directions.
    z : float
        The axial position in the z direction.

    Returns
    -------
    Array
        The Green's function evaluated at the given illumination angle.
    """
    xx, yy = jnp.meshgrid(
        jnp.arange(-params.aperturesize_px, params.aperturesize_px + 1),
        jnp.arange(-params.aperturesize_px, params.aperturesize_px + 1),
        indexing="ij",
    )
    ux = xx + u_shift[0]
    uy = yy + u_shift[1]
    uz_squared = params.k_per_px**2 - ux**2 - uy**2
    mask = uz_squared > 0  # Ensure kz is real
    uz = jnp.sqrt(uz_squared)
    uz = uz * mask  # Set imaginary parts to zero where uz_squared < 0  # noqa: PLR6104

    # Avoid division by zero
    uz_safe = jnp.where(jnp.abs(uz) < _EPSILON, _EPSILON, uz)

    return jnp.exp(-1j * uz_safe * z) / uz_safe


def make_pupil_func(params: IDTParameters, u_shift: tuple[float, float]) -> Array:
    """Create pupil function for IDT reconstruction.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters containing aperture size information
    u_shift : tuple[float, float]
        Shift in frequency domain

    Returns
    -------
    Array
        Pupil function as a disk-shaped mask
    """
    # Convert from fftshift coordinates (-aperturesize_px to +aperturesize_px)
    # to array indices (0 to 2*aperturesize_px)
    center_x = int(u_shift[0] + params.aperturesize_px)
    center_y = int(u_shift[1] + params.aperturesize_px)
    return make_disk((center_x, center_y), params.aperturesize_px / 2, 2 * params.aperturesize_px + 1)


def transfer_func_re(
    params: IDTParameters, u_illumination: tuple[float, float], z: float, incident_intensity: float
) -> Array:
    """Compute real part of transfer function for IDT.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters
    u_illumination : tuple[float, float]
        Illumination angle in frequency domain
    z : float
        Depth position
    incident_intensity : float
        Incident intensity

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
    u_ill_z_squared = params.k_per_px**2 - u_ill_x**2 - u_ill_y**2
    if u_ill_z_squared < 0:
        msg = f"Invalid illumination angle {u_illumination}: u_ill_z_squared must be non-negative."
        raise ValueError(msg)
    u_ill_z = (params.k_per_px**2 - u_ill_x**2 - u_ill_y**2) ** 0.5
    first_term = (
        make_green_func(params, (-u_ill_x, -u_ill_y), z)
        * jnp.exp(-1j * u_ill_z * z)
        * make_pupil_func(params, (-u_ill_x, -u_ill_y))
    )  # maybe first pupil is not correct
    second_term = (
        jnp.conjugate(make_green_func(params, (u_ill_x, u_ill_y), z))
        * jnp.exp(1j * u_ill_z * z)
        * jnp.transpose(jnp.conjugate(make_pupil_func(params, (u_ill_x, u_ill_y))))
    )

    return 1j * params.k_per_px**2 / 2 * incident_intensity * (first_term - second_term)


def transfer_func_im(
    params: IDTParameters, u_illumination: tuple[float, float], z: float, incident_intensity: float
) -> Array:
    """Compute imaginary part of transfer function for IDT.

    Parameters
    ----------
    params : IDTParameters
        IDT parameters
    u_illumination : tuple[float, float]
        Illumination angle in frequency domain
    z : float
        Depth position
    incident_intensity : float
        Incident intensity

    Returns
    -------
    Array
        Imaginary part of transfer function
    """
    u_ill_x, u_ill_y = u_illumination
    u_ill_z = (params.k_per_px**2 - u_ill_x**2 - u_ill_y**2) ** 0.5
    first_term = (
        jnp.conjugate(make_pupil_func(params, (-u_ill_x, -u_ill_y)))
        * make_green_func(params, (-u_ill_x, -u_ill_y), z)
        * jnp.exp(-1j * u_ill_z * z)
        * make_pupil_func(params, (-u_ill_x, -u_ill_y))
    )
    second_term = (
        make_pupil_func(params, (-u_ill_x, -u_ill_y))
        * jnp.conjugate(make_green_func(params, (u_ill_x, u_ill_y), z))
        * jnp.exp(1j * u_ill_z * z)
        * jnp.transpose(jnp.conjugate(make_pupil_func(params, (u_ill_x, u_ill_y))))
    )

    return -(params.k_per_px**2) / 2 * incident_intensity * (first_term + second_term)


def compute_permitivity(
    params: IDTParameters,
    g_tilde_list: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
    led_illumination_intensities: Sequence[Array],
    z: float = 0.0,
    alpha: float = 1e-6,
    beta: float = 1e-6,
) -> tuple[Array, Array]:
    """Compute the permittivity changes Δε_Re and Δε_Im for each slice.

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    g_tilde_list : list of [Nx, Ny] arrays
        Fourier Transforms of the intensity differences per angle.
    u_illumination_list : list of tuples
        The illumination angles for each image.
    led_illumination_intensities : list of [Nx, Ny] arrays
        The incident intensities for each illumination angle.
    z : float, optional
        The axial position in the z direction, by default 0.0
    alpha : float, optional
        Regularization parameter for the real part of the permittivity, by default 1e-6
    beta : float, optional
        Regularization parameter for the imaginary part of the permittivity, by default 1e-6

    Returns
    -------
    tuple[Array, Array]
        The computed permittivity changes Δε_Re and Δε_Im for each slice.
    """
    h_normalized_re = jnp.stack(
        [
            transfer_func_re(params, u_illumination_list[i], z, float(led_illumination_intensities[i]))
            for i in range(len(u_illumination_list))
        ],
        axis=-1,
    )
    h_normalized_im = jnp.stack(
        [
            transfer_func_im(params, u_illumination_list[i], z, float(led_illumination_intensities[i]))
            for i in range(len(u_illumination_list))
        ],
        axis=-1,
    )

    g_tilde = jnp.stack(g_tilde_list, axis=-1)

    sum_h_normalized_re = jnp.sum(jnp.abs(h_normalized_re) ** 2, axis=-1)
    sum_h_normalized_im = jnp.sum(jnp.abs(h_normalized_im) ** 2, axis=-1)

    eps_re_first_term = (sum_h_normalized_im + beta) * jnp.sum(jnp.conjugate(h_normalized_re) * g_tilde, axis=-1)
    eps_re_second_term = jnp.sum(jnp.conjugate(h_normalized_re) * h_normalized_im, axis=-1) * jnp.sum(
        jnp.conjugate(h_normalized_im) * g_tilde, axis=-1
    )

    eps_im_first_term = (sum_h_normalized_re + alpha) * jnp.sum(jnp.conjugate(h_normalized_im) * g_tilde, axis=-1)
    eps_im_second_term = jnp.sum(jnp.conjugate(h_normalized_im) * h_normalized_re, axis=-1) * jnp.sum(
        jnp.conjugate(h_normalized_re) * g_tilde, axis=-1
    )

    # Fix scale_factor calculation to be real-valued and numerically stable
    term1 = (sum_h_normalized_re + alpha) * (sum_h_normalized_im + beta)
    term2 = jnp.abs(jnp.sum(jnp.conjugate(h_normalized_re) * h_normalized_im, axis=-1)) ** 2
    scale_factor = term1 - term2

    # Regularize scale_factor to avoid division by zero
    scale_factor = jnp.where(jnp.abs(scale_factor) < _EPSILON, _EPSILON, scale_factor)

    eps_re = (eps_re_first_term - eps_re_second_term) / scale_factor
    eps_im = (eps_im_first_term - eps_im_second_term) / scale_factor

    return eps_re, eps_im


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
    n_complex = jnp.sqrt(1 + eps_complex)
    n_re = jnp.real(n_complex) * n_sol
    n_im = jnp.imag(n_complex) * n_sol
    return n_re, n_im


##################################################
# Main IDT computation function
##################################################


def compute_idt(
    params: IDTParameters,
    intensity_images: Sequence[Array],
    ref_intensity_images: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
) -> tuple[Array, Array]:
    """Compute the refractive index from intensity images using IDT.

    Outline
    -------
    Step 1: Collect intensity images under different angles
            I_list = [I_1, I_2, …]
            illum_angles = [u_1, u_2,…]
    Step 2: Subtract background and normalize
            g_l = (I_l - I_background) / I_background
            g_list = [g_1, g_2, …]
    Step 3: Fourier Transform each image
            g_tilde_l = fft2(g_l)
            g_tilde_list = [g_tilde_1, …]
    Step 4: Build Transfer Functions, depends on slice depth (z) and the angel of illumination
            H_Re[l, m, x, y] #phase
            H_Im[l, m, x, y] #absorption
    Step 5: Solve inverse problem
            Δε_Re[m] = ifft( weighted_sum_over_l( H_Re_conj * g̃ ) / (|H_Re|² + alpha) )
            Δε_Im[m] = same thing but with H_Im and β
    Slice by slice reconstruct
            Δε_Re[x, y, z]
            Δε_Im[x, y, z]
    Step 6: Convert permittivity to refractive index

    Parameters
    ----------
    params : IDTParameters
        The parameters for the IDT model.
    intensity_images : Sequence[Array]
        The intensity images to process.
    ref_intensity_images : Sequence[Array]
        The reference intensity images for comparison.
    u_illumination_list : Sequence[tuple[float, float]]
        The illumination angles for each LED.

    Returns
    -------
    tuple[Array, Array]
        Real and imaginary parts of the computed refractive index.

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

    # STEP 1: intensity images (input)

    # STEP 2: compute g_l

    g_l = compute_g_list(intensity_images, ref_intensity_images, normalize=True)

    # STEP 3: Fourier Transform each image

    g_tilde_list = fourier_transform(g_l)

    # STEP 4: Build Transfer Functions

    # done in the separate functions

    # STEP 5: Solve inverse problem

    alpha = 1e-2  # Increased regularization parameter for real part
    beta = 1e-2  # Increased regularization parameter for imaginary part

    aperture_size = 2 * params.aperturesize_px + 1
    eps_re_3d = jnp.zeros((aperture_size, aperture_size, params.num_z_slices))
    eps_im_3d = jnp.zeros((aperture_size, aperture_size, params.num_z_slices))

    for z in range(params.num_z_slices):
        eps_re, eps_im = compute_permitivity(
            params,
            g_tilde_list,
            u_illumination_list,
            [jnp.array(1.0) for _ in range(len(u_illumination_list))],  # Assuming uniform intensity for simplicity
            z=z,
            alpha=alpha,
            beta=beta,
        )
        eps_re_3d = eps_re_3d.at[:, :, z].set(eps_re)
        eps_im_3d = eps_im_3d.at[:, :, z].set(eps_im)

    # STEP 6: Convert permittivity to refractive index
    n_re_freq, n_im_freq = convert_to_refractive_index(eps_re_3d, eps_im_3d, params.n_sol)

    # STEP 7: Transform from frequency domain to spatial domain
    # The computed refractive index is in frequency domain, need inverse FFT with ifftshift
    n_re_spatial = jnp.fft.ifft2(jnp.fft.ifftshift(n_re_freq, axes=(0, 1)), axes=(0, 1), norm="ortho")
    n_im_spatial = jnp.fft.ifft2(jnp.fft.ifftshift(n_im_freq, axes=(0, 1)), axes=(0, 1), norm="ortho")

    # Take real part since result should be real in spatial domain
    n_re_spatial = jnp.real(n_re_spatial)
    n_im_spatial = jnp.real(n_im_spatial)

    return n_re_spatial, n_im_spatial
