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
    η: float = (k ** 2 - abs(ui) ** 2) ** (1 / 2)
    L: int
    M: int

#ValueError Nx=Ny must be true
#load images


#convert all I_m to float32 and normalize each image
#Pupil function - P(u)
def make_pupil(Nx, Ny, NA, wavelength_m, px_size_m) -> Array:
    """
    Defines P(u) the pupil in equations.
    Represents the slightly shifted window each LED angle gives shifted into Fourier space.
    """
    kx = jnp.fft.fftfreq(Nx, Ny, d=px_size_m)
    ky = jnp.fft.fftfreq(Nx, Ny, d=px_size_m)
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    freq_radius = jnp.sqrt(KX**2 + KY**2)
    freq_per_px = 1 / (px_size_m * Nx)

    cutoff = NA / wavelength_m / freq_per_px
    P = (freq_radius <= cutoff).astype(jnp.float32)
    return P
    #offaixs shift of pupil is detmerined by ui

    #a pair of shifted pupils shiftig to opposite directions are super-imposed...
    #in the TFs (twin-image holography) illustrated by computed phase and absorption TFs


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

def compute_g_list(I_list: Sequence[Array], Ii:jnp.ndarray, normalize: bool = True) -> list[Array]:
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

# build_transfer_functions():
#L: number of images, M: number of slices, Nx, Ny: image size
Δε_Re = []
Δε_Im = []

def g_tilde_l(g_l):
    return jnp.fft.fft2(g_l, norm"ortho")

def reconstruct_spectrum(g_list, H_list):
    """
    Reconstructs the 3D scattering spectrum Δε from g_list and H_list

    Parameters
    ----------
    g_list : list of 2D arrays (Nx, Ny)
        Measured intensity differences per angle (spatial domain)
    H_list : list of 3D arrays (Nx, Ny, Nz)
        Transfer functions for each angle (frequency domain)

    Returns
    -------
    delta_eps_k : 3D array (Nx, Ny, Nz)
        Estimated scattering potential in Fourier space
    """
for m in range(M):  # for each depth slice

for l in range(L):  # for each illumination angle
    H_Re_conj = jnp.conj(H_Re_lm)
    H_Im_conj = jnp.conj(H_Im_lm)

    # Regularization terms
    alpha = 1e-3
    beta = 1e-3
    #axial direction regulation term is 4 * na / wavelength_m
    #axial elongation in Fourier is up to (2 - 2 * (1 - na **2) ** (1/2)) / wavelength_m

assert len(g_list) == len(H_list)
l_angles = len(g_list)

#get shape
Nx, Ny = g_list[0].shape
Nz = H_list[0].shape[2]

def transfer_functions(kx, ky, kz, illum_angles, k) -> list[jnp.ndarray]
    """
       Generate a list of 3D transfer functions H_l(kx, ky, kz)
    for each illumination angle.

    Parameters
    ----------
    kx, ky, kz : 3D arrays
        Meshgrids of Fourier space coordinates
    illumination_vectors : list of tuples (kx, ky)
        Incident angle vectors in k-space
    k : `float`
        Wavenumber = 2 * math.pi / wavelength_m

    Returns
    -------
    H_list : list of 3D arrays
        Transfer function for each illumination angle
    """
    H_list = []
    for (kx, ky) in illum_angles:
        #scattered wavevector z-component
        kz = jnp.sqrt (k ** 2 - kx ** 2 - ky** 2)
        H_l = (1 / (2 * kz)) * jnp.exp(-1j * kz)
        H_l = k ** 2 / (2 * kz) * make_pupil
        H_list.append(H_l)
    return H_list

H_Re = jnp.zeros((L, M, Nx, Ny), dtype=complex)
H_Im = jnp.zeros_like(H_Re)

for l, tran_freq in enumerate(illum_angles):
    for m , z in enumerate(z_vals) for each slice z:
        H_Re[l, m] = compute H_Re(u, z | ui) using Eq. (5);
        H_Im[l, m] = compute H_Im(u, z | ui) using Eq. (6);
        normalize Hs by dividing by Ii
        store TFs

arr = jnp.Array([])
arr_conj = arr.conjugate()
arr_conj = jnp.conj(arr)
H_Re_conj = H_Re.conjugate()
H_Im_conj = H_Im.conjugate()


#deconvolve_slice (H_Re, H_Im, g_tilde, alpha, beta)
for each slice m:
    A = sum(|H_Re| ** 2 + α) * sum(|H_Im| ** 2 + β) - ((sum(H_Re * H_Im_conj))*(sum(H_Re_conj*H_Im)))
    Δε_Re[m] = jnp.fft.ifft2 ( (1/A) * [sum(H_Re_conj *  g_tilde) - (sum(H_Re_conj * H_Im) * (sum(H_Im_conj * g_tilde)))] )
    Δε_Im[m] = jnp.fft.ifft2 ( (1/A) * [sum(H_Im_conj * g_tilde) - (sum(H_Re * H_Im_conj) * (sum(H_Re_conj * g_tilde)))] )
#loop over all m slices


Δε_Re_3D = jnp.stack(Δε_Re[m] for all m, axis=-1)
Δε_Im_3D = jnp.stack(Δε_Im[m] for all m, axis=-1)

n_sol = 1.33
n = jnp.sqrt(n_sol ** 2 + Δε_Re_3D(x, y, z))
#n(x, y, z) = sqrt(ε₀ + Δε_Re_3D(x, y, z))  relation between refractive index and permittivity contrast

#generate_pupil(params) and apply_apodization()

"""
Output
------

Δε_Re_3D: phase
Δε_Im_3D: absorption
n(x,y, z): refractive index map
"""
