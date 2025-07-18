"""Intensity Diffraction Tomography (IDT) module."""

#Start IDT psuedocode

#Download packages like

"""import dataclasses
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from tqdm import tqdm
and Muscopy"""
import dataclasses
from typing import TYPE_CHECKING
import jax.numpy as jnp
from jax import Array
from tqdm import tqdm

from muscopy.dh import MuParameters, make_disk
from muscopy.cfg import OffsetRegions, ArrayPrecision
from muscopy.qpi_utils import unwrap_phase

"""
Outline
-------
Step 1: Collect intensity images under different angles
	I_list = [I_1, I_2, …]
	illum_angles = [ui_1, ui_2,…]
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
	Δε_Re[m] = ifft( weighted_sum_over_l( H_Re_conj × g̃ ) / (|H_Re|² + α) )
	Δε_Im[m] = same thing but with H_Im and β
Slice by slice reconstruct 
	Δε_Re[x, y, z] 
	Δε_Im[x, y, z]
Step 6: Convert permittivity to refractive index

"""
@dataclasses.dataclass
class IDTParameters(MuParameters):
    """
    Intensity Diffraction Tomography (IDT) Parameters:

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

    def make_pupil(Nx, Ny, NA, wavelength_m, px_size):
        """
        Defines P(u) the pupil in equations.
        Represents the slightly shifted window each LED angle gives shifted into Fourier space.
        """
        kx = jnp.fft.fftfreq(Nx, d=px_size)
        ky = jnp.fft.fftfreq(Ny, d=px_size)
        KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
        freq_radius = jnp.sqrt(KX**2 + KY**2)

        cutoff = NA / wavelength_m
        P = (freq_radius <= cutoff).astype(np.float32)
        return P
    

   #find permittivity contrast by e = (n ** 2) / (mu) relation between permittivity contrast and refractive index
   # k = 2 * math.pi / wavelength_m
   #ui - illumination angle (transverse frequency)
   #η(ui) - axial spatial frequency
   #P(u) - Pupil function
   #I(x) - Measured intensity

   #I_list: List of 2D intensity images
   #ui_list: List of corresponding illumination angles
   #params: imaging parameters (Muparameters)
   #config: Regularization and reconstruction config

for I in I_list:
    Ii = estimate_background(I)
    g = (I - Ii) / Ii
    store(g) or g_list.append(g)



H_Re = jnp.zeros((L, M, Nx, Ny), dtype=complex)
H_Im = jnp.zeros_like(H_Re)

for l, ui in enumerate(illum_angles):
    for m , z in enumerate(z_vals) for each slice z:
        H_Re[l, m] = compute H_Re(u, z | ui) using Eq. (5)
        H_Im[l, m] = compute H_Im(u, z | ui) using Eq. (6)
        normalize Hs by dividing by Ii
        store TFs

g_tilde = jnp.fft.fft2(g_list, norm"ortho")
#Shape: (L, Nx, Ny)


arr = jnp.Array([])
arr_conj = arr.conjugate()
arr_conj = jnp.conj(arr)

#deconvolve_slice (H_Re, H_Im, g_tilde, alpha, beta)
for each slice m:
    A = ∑|H_Re| ** 2 + α * ∑|H_Im| ** 2 + β - cross_terms
    Δε_Re[m] = jnp.fft.ifft2 ( (1/A) * [sum(H_Re_conj *  g_tilde) - cross_term] )
    Δε_Im[m] = jnp.fft.ifft2 ( (1/A) * [sum(H_Im_conj * g_tilde) - cross_term] )
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
