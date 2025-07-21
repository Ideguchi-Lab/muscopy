"""Intensity Diffraction Tomography (IDT) module."""

import dataclasses
import math
from typing import TYPE_CHECKING, Sequence
import jax.numpy as jnp
from jax import Array
from tqdm import tqdm
from __future__ import annotations


from muscopy.dh import MuParameters, make_disk
from muscopy.cfg import OffsetRegions, ArrayPrecision
from muscopy.qpi_utils import unwrap_phase
from muscopy.dir_parser import numpy_parser

"""
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
	Δε_Re[m] = ifft( weighted_sum_over_l( H_Re_conj * g̃ ) / (|H_Re|² + α) )
	Δε_Im[m] = same thing but with H_Im and β
Slice by slice reconstruct
	Δε_Re[x, y, z]
	Δε_Im[x, y, z]
Step 6: Convert permittivity to refractive index

"""


@dataclasses.dataclass
class IDTParameters:
    """
    Intensity Diffraction Tomography (IDT) Parameters:

    Attributes
    ----------
    na : `float`
        Numerical aperture of the objective lens
    wavelength_m : `float`
        Wavelength of the light in meters
    Nx : `int`
        Size of the image in pixels. We assume square image and = Ny
    Ny : `int`
        Size of the image in pixelss.
    px_size_m : `float`
         Pixel size in meters
    n_sol : `float`
        Refractive index of the solution
    na_illumination : `float`
        Maximum illumination numerical aperture
    I_list : `tuple`
        Difference intensity images
    illum_angles : `tuple`
        The different angles of the intensity image
    LED_i : `float`
        Intensity of the ith LED (formally S_i)
    k : `int`
        Wavenumber 2 * math.pi / wavelength_m
    ui : `int`
        transverse frequency
    η : `int`
        axial spatial frequency
    L : `int`
        number of illumination angles
    M : `int`
        number of slices
    """

    na: float
    wavelength_m: float
    Nx: int
    Ny: int
    px_size_m: float
    n_sol: float
    na_illumination: float
    I_list: tuple
    illum_angles: tuple
    LED_i: float
    k: float = 2 * math.pi / wavelength_m
    ui: int
    η: float = (k**2 - abs(ui) ** 2) ** (1 / 2)
    L: int
    M: int

    @property
    def aperturesize_px(self) -> int:
        return 2 * round(self.na / self.wavelength_m / self.freq_per_px) + 1

    @property
    def freq_per_px(self) -> float:
        """Frequency per pixel in Fourier space."""
        return 1 / (self.px_size_m * self.Nx)


# ValueError Nx=Ny must be true
# load images


# convert all I_m to float32 and normalize each image
# Pupil function - P(u)
def make_pupil(Nx, Ny, NA, wavelength_m, px_size_m) -> Array:
    """
    Defines P(u) the pupil in equations.
    Represents the slightly shifted window each LED angle gives shifted into Fourier space.
    """
    kx = jnp.fft.fftfreq(Nx, Ny, d=px_size_m)
    ky = jnp.fft.fftfreq(Nx, Ny, d=px_size_m)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    freq_radius = jnp.sqrt(KX**2 + KY**2)
    freq_per_px = 1 / (px_size_m * Nx)

    cutoff = NA / wavelength_m / freq_per_px
    P = (freq_radius <= cutoff).astype(jnp.float32)
    return P
    # offaixs shift of pupil is detmerined by ui

    # a pair of shifted pupils shiftig to opposite directions are super-imposed...
    # in the TFs (twin-image holography) illustrated by computed phase and absorption TFs


##################################################
# Step 1: Collect Intensity Images
##################################################

# which can be generated from Multi-layer Born simulator or experiments
data_path = "path_to_data"
bg_path = "path_to_background"

data_images_path = numpy_parser(data_path)
I_list = [jnp.load(image_path) for image_path in data_images_path]
Ii = jnp.load(bg_path)

##################################################
# Step 2: Subtract and Normalize
###################################################


def compute_g_list(I_list: Sequence[Array], Ii: jnp.ndarray, normalize: bool = True) -> list[Array]:
    """Computes list of intensity constrasts g_l for each illumination angle.

    Parameters
    ----------
    I_list : list of [Nx, Ny] arrays of Intensity images
        under different angles
    Ii : background intensity image for subtraction
    normalize : bool
        whether or not to normalize

    Returns
    -------
    g_list : list of [Nx, Ny] arrays
        Intensity different or contrast for each angle.
    """
    g_list = []
    for I_m in I_list:
        if normalize:
            g = (I_m - Ii) / Ii
            g_list.append(g)
    return g_list


g_list = compute_g_list(I_list, Ii, normalize=True)

##################################################
# Step 3: Fourier Transform each image
##################################################


def fourier_transform(g_list: Sequence[Array]) -> list[Array]:
    """
    Computes the Fourier Transform of each g_l in g_list.

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


g_tilde_list = fourier_transform(g_list)

##################################################
# Step 4: Build Transfer Functions
##################################################


def make_green_func(params: IDTParameters, u_shift: tuple[float, float], z: float) -> Array:
    """Generates the Green's function for a given illumination angle.

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
    uz_squared = params.k**2 - ux**2 - uy**2
    mask = uz_squared > 0  # Ensure kz is real
    uz = jnp.sqrt(uz_squared)
    uz = uz * mask  # Set imaginary parts to zero where uz_squared < 0

    return jnp.exp(-1j * uz * z) / uz


def make_pupil_func(params: IDTParameters, u_shift: tuple[float, float]) -> Array:
    return make_disk(u_shift, params.aperturesize_px / 2, 2 * params.aperturesize_px + 1)


def transfer_func_re(
    params: IDTParameters, u_illumination: tuple[float, float], z: float, incident_intensity: float
) -> Array:
    u_ill_x, u_ill_y = u_illumination
    u_ill_z = (params.k**2 - u_ill_x**2 - u_ill_y**2) ** 0.5
    first_term = (
        jnp.conjugate(make_pupil_func(params, (-u_ill_x, -u_ill_y)))
        * make_green_func(params, (-u_ill_x, -u_ill_y), z)
        * jnp.exp(-1j * u_ill_z * z)
        * make_pupil_func(params, (-u_ill_x, -u_ill_y))
    )  # maybe first pupil is not correct
    second_term = (
        make_pupil_func(params, (-u_ill_x, -u_ill_y))
        * jnp.conjugate(make_green_func(params, (u_ill_x, u_ill_y), z))
        * jnp.exp(1j * u_ill_z * z)
        * jnp.transpose(jnp.conjugate(make_pupil_func(params, (u_ill_x, u_ill_y))))
    )

    return 1j * params.k**2 / 2 * incident_intensity * (first_term - second_term)


def transfer_func_im(
    params: IDTParameters, u_illumination: tuple[float, float], z: float, incident_intensity: float
) -> Array:
    u_ill_x, u_ill_y = u_illumination
    u_ill_z = (params.k**2 - u_ill_x**2 - u_ill_y**2) ** 0.5
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

    return -(params.k**2) / 2 * incident_intensity * (first_term + second_term)


"""
Output
------

Δε_Re_3D: phase
Δε_Im_3D: absorption
n(x,y, z): refractive index map
"""
