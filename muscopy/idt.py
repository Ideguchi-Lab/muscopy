"""Intensity Diffraction Tomography (IDT) module."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from muscopy.dh import make_disk
from muscopy.dir_parser import numpy_parser
from muscopy.odt import ODTParameters

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclasses.dataclass
class IDTParameters(ODTParameters):
    """IDT Parameters."""

    num_z_slices: int


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
            g = (i_m - i_ref) / i_ref
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
        g_tilde = jnp.fft.fft2(g_l, norm="ortho")  # FFT with orthonormal normalization
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
        jnp.arange(-params.aperturesize_px, params.aperturesize_px),
        jnp.arange(-params.aperturesize_px, params.aperturesize_px),
        indexing="ij",
    )
    ux = xx + u_shift[0]
    uy = yy + u_shift[1]
    uz_squared = params.k_per_px**2 - ux**2 - uy**2
    mask = uz_squared > 0  # Ensure kz is real
    uz = jnp.sqrt(uz_squared)
    uz = uz * mask  # Set imaginary parts to zero where uz_squared < 0  # noqa: PLR6104

    return jnp.exp(-1j * uz * z) / uz


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
    return make_disk(u_shift, params.aperturesize_px / 2, 2 * params.aperturesize_px + 1)


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
            transfer_func_re(params, u_illumination_list[i], z, led_illumination_intensities[i])
            for i in range(len(u_illumination_list))
        ],
        axis=-1,
    )
    h_normalized_im = jnp.stack(
        [
            transfer_func_im(params, u_illumination_list[i], z, led_illumination_intensities[i])
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

    scale_factor = (h_normalized_re + alpha) * (h_normalized_im + beta) - jnp.sum(
        jnp.conjugate(h_normalized_re) * h_normalized_im, axis=-1
    ) * jnp.sum(jnp.conjugate(h_normalized_im) * h_normalized_re, axis=-1)

    eps_re = (eps_re_first_term - eps_re_second_term) / scale_factor
    eps_im = (eps_im_first_term - eps_im_second_term) / scale_factor

    return eps_re, eps_im


def convert_to_refractive_index(eps_3d: Array, n_sol: float) -> tuple[Array, Array]:
    """Convert the difference in permittivity to difference in refractive index.

    Parameters
    ----------
    eps_3d : `Array`
        3D array of permittivity values
    n_col : `float`
        Refractive index of the solution

    Returns
    -------
    n_complex : tuple[Array, Array]
        Real and Imaginary components of the refractive index difference

    """
    n_complex = jnp.sqrt(eps_3d)
    return n_complex * n_sol


##################################################
# Main IDT computation function
##################################################


def compute_idt(
    params: IDTParameters,
    intensity_images: Sequence[Array],
    ref_intensity_images: Sequence[Array],
    u_illumination_list: Sequence[tuple[int, int]],
) -> Array:
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
    u_illumination_list : Sequence[tuple[int, int]]
        The illumination angles for each LED.

    Returns
    -------
    Array
        The computed refractive index.

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

    alpha = 1e-6  # Regularization parameter for real part
    beta = 1e-6  # Regularization parameter for imaginary part

    eps_3d = jnp.zeros((params.img_size_px, params.img_size_px, params.num_z_slices))

    for z in range(params.num_z_slices):
        eps_re, eps_im = compute_permitivity(
            params,
            g_tilde_list,
            u_illumination_list,
            [1.0] * len(u_illumination_list),  # Assuming uniform intensity for simplicity
            z=z,
            alpha=alpha,
            beta=beta,
        )
        eps_3d[:, :, z] = eps_re + 1j * eps_im

    # STEP 6: Convert permittivity to refractive index

    return convert_to_refractive_index(eps_3d, params.n_sol)
