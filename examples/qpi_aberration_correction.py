"""
Demonstration of QPI reconstruction with aberration correction
==============================================================

This example demonstrates how to use the `muscopy.dh.correct_aberration` function
to correct phase aberrations in the transfer function during QPI reconstruction.
The example compares reconstruction with and without aberration correction.
"""

# %%
# import modules

import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, correct_aberration, get_spectrum, make_disk, offaxis_dh, print_all_parameters
from muscopy.qpi import correct_phase_offset

# config
SHOW_IMAGE = True

print(jax.default_backend())

# %%
# set Microscopy parameters

params = MuParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=512,  # Reduced size for faster computation
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)

# show QPI parameters
print_all_parameters(params, show_properties=True)

# %%
# Create aberration in the complex amplitude

# Create coordinate grids for the full image
xx_full, yy_full = jnp.meshgrid(
    jnp.arange(params.img_size_px) - params.img_center[0],
    jnp.arange(params.img_size_px) - params.img_center[1],
    indexing="ij",
)

# Normalize coordinates to the aperture size
rho_full = jnp.sqrt(xx_full**2 + yy_full**2) / (params.aperturesize_px // 2)
theta_full = jnp.arctan2(yy_full, xx_full)

# Create aberration: combination of spherical aberration and astigmatism
# Spherical aberration: rho^4 term
spherical_aberration_full = 0.5 * rho_full**4
# Astigmatism: rho^2 * cos(2*theta) term
astigmatism_full = 0.3 * rho_full**2 * jnp.cos(2 * theta_full)
# Coma: rho^3 * cos(theta) term
coma_full = 0.2 * rho_full**3 * jnp.cos(theta_full)

# Total aberration phase (in radians)
aberration_phase_full = spherical_aberration_full + astigmatism_full + coma_full

# Apply aberration only within the aperture
aperture_mask_full = rho_full <= 1.0
aberration_function = jnp.where(aperture_mask_full, jnp.exp(1j * aberration_phase_full), 1.0)

pupil_function = aberration_function[
    params.img_center[0] - params.aperturesize_px // 2 : params.img_center[0] + params.aperturesize_px // 2 + 1,
    params.img_center[1] - params.aperturesize_px // 2 : params.img_center[1] + params.aperturesize_px // 2 + 1,
]

# %%
# create a hologram
off_axis_center = (150, 150)

radius = 30
sample_disk = make_disk(params.img_center, radius, params.img_size_px)
sample_array = jnp.exp(2j * math.pi / 3 * sample_disk)
low_pass = make_disk(params.img_center, params.aperturesize_px // 2, params.img_size_px)
ft_sample = jnp.fft.fftshift(jnp.fft.fft2(sample_array))
ft_sample_aberration = ft_sample * aberration_function
sample_array = jnp.fft.ifft2(jnp.fft.ifftshift(ft_sample))
sample_array /= jnp.sum(jnp.abs(sample_array) ** 2) ** 0.5

sample_array_aberration = jnp.fft.ifft2(jnp.fft.ifftshift(ft_sample_aberration))
sample_array_aberration /= jnp.sum(jnp.abs(sample_array_aberration) ** 2) ** 0.5

xx, yy = jnp.meshgrid(
    jnp.arange(params.img_size_px),
    jnp.arange(params.img_size_px),
    indexing="ij",
)
ref_array = jnp.exp(
    -2j
    * jnp.pi
    * (
        xx * (off_axis_center[0] - params.img_center[0]) / (params.img_size_px)
        + yy * (off_axis_center[1] - params.img_center[1]) / (params.img_size_px)
    )
)
ref_array /= jnp.sum(jnp.abs(ref_array) ** 2) ** 0.5

hologram = jnp.abs(sample_array + ref_array) ** 2
hologram_aberration = jnp.abs(sample_array_aberration + ref_array) ** 2

ref_sample_array = jnp.ones_like(sample_array)
ref_sample_array /= jnp.sum(jnp.abs(ref_sample_array) ** 2) ** 0.5
ref_hologram = jnp.abs(ref_sample_array + ref_array) ** 2
ref_hologram_aberration = ref_hologram  # Use same reference for both cases


# %%

# Reconstruct complex field with aberration using offaxis_dh
cp_field_with_aberration = offaxis_dh(hologram_aberration, ref_hologram_aberration, params, off_axis_center)
phase_with_aberration = jnp.angle(cp_field_with_aberration)

# Reconstruct complex field without aberration
cp_field_without_aberration = offaxis_dh(hologram, ref_hologram, params, off_axis_center)
phase_without_aberration = jnp.angle(cp_field_without_aberration)

# %%

# Correct aberration with known pupil function
# Get spectrum from hologram with aberration
hologram_spectrum = get_spectrum(
    jnp.fft.fftshift(jnp.fft.fft2(hologram_aberration)) * params.hologram2spectrum, params, off_axis_center
)
ref_hologram_spectrum = get_spectrum(
    jnp.fft.fftshift(jnp.fft.fft2(ref_hologram_aberration)) * params.hologram2spectrum, params, off_axis_center
)

# Apply aberration correction
hologram_spectrum_corrected = correct_aberration(hologram_spectrum, pupil_function)
ref_hologram_spectrum_corrected = correct_aberration(ref_hologram_spectrum, pupil_function)

# Reconstruct complex field
cp_field_corrected_temp = jnp.fft.ifft2(jnp.fft.ifftshift(hologram_spectrum_corrected)) * params.spectrum2cpfield
ref_cp_field_corrected_temp = (
    jnp.fft.ifft2(jnp.fft.ifftshift(ref_hologram_spectrum_corrected)) * params.spectrum2cpfield
)
cp_field_aberration_corrected = cp_field_corrected_temp / ref_cp_field_corrected_temp

# Extract phase
phase_corrected = jnp.angle(cp_field_aberration_corrected)

# %%
# Apply phase offset correction to both images

offset_regions = [
    ((5, 10), (5, 10)),
    ((params.aperturesize_px - 10, params.aperturesize_px - 5), (5, 10)),
    ((5, 10), (params.aperturesize_px - 10, params.aperturesize_px - 5)),
    (
        (params.aperturesize_px - 10, params.aperturesize_px - 5),
        (params.aperturesize_px - 10, params.aperturesize_px - 5),
    ),
]

phase_with_aberration = correct_phase_offset(phase_with_aberration, offset_regions)
phase_without_aberration = correct_phase_offset(phase_without_aberration, offset_regions)
phase_corrected = correct_phase_offset(phase_corrected, offset_regions)

# %%
# Visualize results

if SHOW_IMAGE:
    # Convert to numpy arrays for visualization
    phase_without_show = jax.device_get(phase_without_aberration)
    phase_with_show = jax.device_get(phase_with_aberration)
    phase_corrected_show = jax.device_get(phase_corrected)
    aberration_phase_show = jax.device_get(jnp.angle(aberration_function))
    pupil_phase_show = jax.device_get(jnp.angle(pupil_function))

    # Set common color scale for phase images
    vmin = min(jnp.min(phase_without_show), jnp.min(phase_with_show), jnp.min(phase_corrected_show))
    vmax = max(jnp.max(phase_without_show), jnp.max(phase_with_show), jnp.max(phase_corrected_show))

    # Create main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Top row: Aberration maps
    # Aberration function applied to full image
    im0 = axes[0, 0].imshow(aberration_phase_show, cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
    axes[0, 0].set_title("Full Image Aberration Phase")
    axes[0, 0].set_xlabel("x [pixels]")
    axes[0, 0].set_ylabel("y [pixels]")
    plt.colorbar(im0, ax=axes[0, 0], label="Phase [rad]")

    # Pupil function (cropped aberration)
    im1 = axes[0, 1].imshow(pupil_phase_show, cmap="twilight", vmin=-jnp.pi, vmax=jnp.pi)
    axes[0, 1].set_title("Pupil Function Phase (Cropped)")
    axes[0, 1].set_xlabel("x [pixels]")
    axes[0, 1].set_ylabel("y [pixels]")
    plt.colorbar(im1, ax=axes[0, 1], label="Phase [rad]")

    # Difference between with/without aberration
    diff_original = phase_with_show - phase_without_show
    im2 = axes[0, 2].imshow(diff_original, cmap="RdBu_r")
    axes[0, 2].set_title("Phase Difference\n(With Aberration - Without)")
    axes[0, 2].set_xlabel("x [pixels]")
    axes[0, 2].set_ylabel("y [pixels]")
    plt.colorbar(im2, ax=axes[0, 2], label="Phase Difference [rad]")

    # Bottom row: Reconstructed phase images
    # Without aberration (reference)
    im3 = axes[1, 0].imshow(phase_without_show, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("Phase without Aberration\n(Reference)")
    axes[1, 0].set_xlabel("x [pixels]")
    axes[1, 0].set_ylabel("y [pixels]")
    plt.colorbar(im3, ax=axes[1, 0], label="Phase [rad]")

    # With aberration
    im4 = axes[1, 1].imshow(phase_with_show, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("Phase with Aberration\n(Degraded)")
    axes[1, 1].set_xlabel("x [pixels]")
    axes[1, 1].set_ylabel("y [pixels]")
    plt.colorbar(im4, ax=axes[1, 1], label="Phase [rad]")

    # Corrected aberration
    im5 = axes[1, 2].imshow(phase_corrected_show, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 2].set_title("Phase after Correction\n(Restored)")
    axes[1, 2].set_xlabel("x [pixels]")
    axes[1, 2].set_ylabel("y [pixels]")
    plt.colorbar(im5, ax=axes[1, 2], label="Phase [rad]")

    plt.tight_layout()
    plt.show()

    # Create detailed comparison figure
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))

    # Difference maps
    diff_corrected = phase_corrected_show - phase_without_show
    diff_improvement = phase_with_show - phase_corrected_show

    # Corrected vs Reference
    im6 = axes2[0, 0].imshow(diff_corrected, cmap="RdBu_r")
    axes2[0, 0].set_title("Correction Error\n(Corrected - Reference)")
    axes2[0, 0].set_xlabel("x [pixels]")
    axes2[0, 0].set_ylabel("y [pixels]")
    plt.colorbar(im6, ax=axes2[0, 0], label="Phase Difference [rad]")

    # Improvement map
    im7 = axes2[0, 1].imshow(diff_improvement, cmap="RdBu_r")
    axes2[0, 1].set_title("Correction Improvement\n(With Aberration - Corrected)")
    axes2[0, 1].set_xlabel("x [pixels]")
    axes2[0, 1].set_ylabel("y [pixels]")
    plt.colorbar(im7, ax=axes2[0, 1], label="Phase Improvement [rad]")

    # Line profiles through center
    center_idx = params.aperturesize_px // 2
    axes2[1, 0].plot(phase_without_show[center_idx, :], label="Without Aberration", linewidth=2, alpha=0.8)
    axes2[1, 0].plot(phase_with_show[center_idx, :], label="With Aberration", linewidth=2, alpha=0.8)
    axes2[1, 0].plot(phase_corrected_show[center_idx, :], label="Corrected", linewidth=2, alpha=0.8)
    axes2[1, 0].set_xlabel("x [pixels]")
    axes2[1, 0].set_ylabel("Phase [rad]")
    axes2[1, 0].set_title("Horizontal Center Line Profile")
    axes2[1, 0].legend()
    axes2[1, 0].grid(True, alpha=0.3)

    # Vertical line profile
    axes2[1, 1].plot(phase_without_show[:, center_idx], label="Without Aberration", linewidth=2, alpha=0.8)
    axes2[1, 1].plot(phase_with_show[:, center_idx], label="With Aberration", linewidth=2, alpha=0.8)
    axes2[1, 1].plot(phase_corrected_show[:, center_idx], label="Corrected", linewidth=2, alpha=0.8)
    axes2[1, 1].set_xlabel("y [pixels]")
    axes2[1, 1].set_ylabel("Phase [rad]")
    axes2[1, 1].set_title("Vertical Center Line Profile")
    axes2[1, 1].legend()
    axes2[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistical analysis
    print("\n" + "=" * 60)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 60)

    # Phase statistics
    print("\n--- Phase Statistics ---")
    print(f"Without Aberration - Mean: {jnp.mean(phase_without_show):.4f}, Std: {jnp.std(phase_without_show):.4f}")
    print(f"With Aberration    - Mean: {jnp.mean(phase_with_show):.4f}, Std: {jnp.std(phase_with_show):.4f}")
    print(f"Corrected          - Mean: {jnp.mean(phase_corrected_show):.4f}, Std: {jnp.std(phase_corrected_show):.4f}")

    # Error metrics
    print("\n--- Error Metrics (RMS) ---")
    rms_aberration_error = jnp.sqrt(jnp.mean(diff_original**2))
    rms_correction_error = jnp.sqrt(jnp.mean(diff_corrected**2))
    rms_improvement = jnp.sqrt(jnp.mean(diff_improvement**2))

    print(f"Aberration Error (With - Without):     {rms_aberration_error:.4f} rad")
    print(f"Correction Error (Corrected - Without): {rms_correction_error:.4f} rad")
    print(f"Improvement (With - Corrected):        {rms_improvement:.4f} rad")

    # Correction efficiency
    correction_efficiency = (rms_aberration_error - rms_correction_error) / rms_aberration_error * 100
    print(f"\nCorrection Efficiency: {correction_efficiency:.1f}%")

    # Peak-to-valley analysis
    print("\n--- Peak-to-Valley Analysis ---")
    pv_without = jnp.max(phase_without_show) - jnp.min(phase_without_show)
    pv_with = jnp.max(phase_with_show) - jnp.min(phase_with_show)
    pv_corrected = jnp.max(phase_corrected_show) - jnp.min(phase_corrected_show)

    print(f"Without Aberration P-V: {pv_without:.4f} rad")
    print(f"With Aberration P-V:    {pv_with:.4f} rad")
    print(f"Corrected P-V:          {pv_corrected:.4f} rad")

    # Signal-to-noise-like ratio
    print("\n--- Signal Quality ---")
    signal_without = jnp.std(phase_without_show)
    noise_aberration = jnp.std(diff_original)
    noise_correction = jnp.std(diff_corrected)

    snr_degraded = 20 * jnp.log10(signal_without / noise_aberration)
    snr_corrected = 20 * jnp.log10(signal_without / noise_correction)

    print(f"SNR with Aberration:    {snr_degraded:.1f} dB")
    print(f"SNR after Correction:   {snr_corrected:.1f} dB")
    print(f"SNR Improvement:        {snr_corrected - snr_degraded:.1f} dB")

    print("\n" + "=" * 60)

# %%
