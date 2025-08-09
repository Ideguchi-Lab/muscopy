"""
Demonstration of Phase Gradient Correction
===========================================

This example demonstrates how to use the `muscopy.dh.correct_gradient` function to correct
phase gradients in complex amplitude data. The function removes linear phase slopes that
may arise from optical misalignments or sample positioning.
"""

# %%
# import modules

import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, correct_gradient, correct_offset, make_disk
from muscopy.qpi_utils import unwrap_phase

# config
SHOW_IMAGE = True

print(jax.default_backend())

# %%
# Set microscopy parameters

params = MuParameters(
    na=0.8,
    wavelength_m=500e-9,
    img_size_px=256,
    px_size_m=5e-6 / 60,
    n_sol=1.33,
)

# %%
# Create a complex amplitude with phase gradient

# Create a simple phase pattern (circular disk)
radius = 30
sample_disk = make_disk(params.img_center, radius, params.img_size_px)
# Use a larger phase shift to make the pattern more visible
base_phase = math.pi * sample_disk.astype(float)  # 0 outside, π inside

# Add linear phase gradient (tilt) - this simulates optical misalignment
xx, yy = jnp.meshgrid(
    jnp.arange(params.img_size_px),
    jnp.arange(params.img_size_px),
    indexing="ij",
)

# Define gradient slopes
gradient_x = 0.02  # phase gradient in x direction [rad/pixel]
gradient_y = 0.015  # phase gradient in y direction [rad/pixel]

# Add linear phase gradient
phase_with_gradient = base_phase + gradient_x * (xx - params.img_center[0]) + gradient_y * (yy - params.img_center[1])

# Create complex amplitude with uniform magnitude
magnitude = jnp.ones_like(phase_with_gradient) * 0.8
complex_amplitude_with_gradient = magnitude * jnp.exp(1j * phase_with_gradient)

# %%
# Apply gradient correction

# First apply gradient correction
complex_amplitude_gradient_corrected = correct_gradient(complex_amplitude_with_gradient)

# Define offset regions for constant phase correction (use corners)
aperture_size = params.aperturesize_px
offset_regions = [
    ((5, 10), (5, 10)),  # top-left
    ((aperture_size - 10, aperture_size - 5), (5, 10)),  # top-right
    ((5, 10), (aperture_size - 10, aperture_size - 5)),  # bottom-left
    ((aperture_size - 10, aperture_size - 5), (aperture_size - 10, aperture_size - 5)),  # bottom-right
]

# Apply constant phase offset correction
complex_amplitude_corrected = correct_offset(
    complex_amplitude_gradient_corrected, offset_regions, phase=True, amplitude=False
)

# %%
# Extract phases for comparison

phase_original = jnp.angle(complex_amplitude_with_gradient)
phase_gradient_corrected = jnp.angle(complex_amplitude_gradient_corrected)
phase_corrected = jnp.angle(complex_amplitude_corrected)

# Calculate phase differences to show the correction
phase_diff = phase_corrected - phase_original

# %%
# Visualize results

if SHOW_IMAGE:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original phase with gradient
    im1 = axes[0, 0].imshow(jax.device_get(phase_original), cmap="hsv", vmin=-jnp.pi, vmax=jnp.pi)
    axes[0, 0].set_title("Original Phase\n(with gradient)")
    axes[0, 0].set_xlabel("x [pixels]")
    axes[0, 0].set_ylabel("y [pixels]")
    plt.colorbar(im1, ax=axes[0, 0], label="Phase [rad]")

    # After gradient correction only
    im2 = axes[0, 1].imshow(jax.device_get(phase_gradient_corrected), cmap="hsv", vmin=-jnp.pi, vmax=jnp.pi)
    axes[0, 1].set_title("After Gradient Correction\n(linear slope removed)")
    axes[0, 1].set_xlabel("x [pixels]")
    axes[0, 1].set_ylabel("y [pixels]")
    plt.colorbar(im2, ax=axes[0, 1], label="Phase [rad]")

    # Final corrected phase (gradient + offset correction)
    im3 = axes[0, 2].imshow(jax.device_get(phase_corrected), cmap="hsv", vmin=-jnp.pi, vmax=jnp.pi)
    axes[0, 2].set_title("Final Corrected Phase\n(circle visible)")
    axes[0, 2].set_xlabel("x [pixels]")
    axes[0, 2].set_ylabel("y [pixels]")
    plt.colorbar(im3, ax=axes[0, 2], label="Phase [rad]")

    # Target phase (ideal reference)
    im4 = axes[1, 0].imshow(jax.device_get(base_phase), cmap="hsv", vmin=-jnp.pi, vmax=jnp.pi)
    axes[1, 0].set_title("Target Phase\n(no gradient)")
    axes[1, 0].set_xlabel("x [pixels]")
    axes[1, 0].set_ylabel("y [pixels]")
    plt.colorbar(im4, ax=axes[1, 0], label="Phase [rad]")

    # Amplitude preservation check
    amplitude_original = jnp.abs(complex_amplitude_with_gradient)
    amplitude_corrected = jnp.abs(complex_amplitude_corrected)
    im5 = axes[1, 1].imshow(jax.device_get(amplitude_corrected), cmap="viridis")
    axes[1, 1].set_title("Amplitude\n(preserved)")
    axes[1, 1].set_xlabel("x [pixels]")
    axes[1, 1].set_ylabel("y [pixels]")
    plt.colorbar(im5, ax=axes[1, 1], label="Amplitude")

    # Correction accuracy with unwrapped phases
    phase_corrected_unwrapped = unwrap_phase(phase_corrected)
    base_phase_unwrapped = unwrap_phase(base_phase)
    correction_error = jnp.abs(phase_corrected_unwrapped - base_phase_unwrapped)
    im6 = axes[1, 2].imshow(jax.device_get(correction_error), cmap="plasma")
    axes[1, 2].set_title("Correction Error\n(unwrapped phases)")
    axes[1, 2].set_xlabel("x [pixels]")
    axes[1, 2].set_ylabel("y [pixels]")
    plt.colorbar(im6, ax=axes[1, 2], label="Error [rad]")

    plt.tight_layout()
    plt.show()

# %%
# Quantitative analysis


# Calculate phase gradients before and after correction
def calculate_gradients(phase: jnp.ndarray) -> tuple[float, float]:
    """Calculate phase gradients in x and y directions.

    Parameters
    ----------
    phase : jnp.ndarray
        Phase array to analyze

    Returns
    -------
    tuple[float, float]
        Mean phase gradients in x and y directions
    """
    grad_x = float(jnp.mean(jnp.diff(phase, axis=1)))
    grad_y = float(jnp.mean(jnp.diff(phase, axis=0)))
    return grad_x, grad_y


grad_x_original, grad_y_original = calculate_gradients(phase_original)
grad_x_corrected, grad_y_corrected = calculate_gradients(phase_corrected)

print("Phase gradient analysis:")
print(f"Original gradients: x = {grad_x_original:.6f} rad/pixel, y = {grad_y_original:.6f} rad/pixel")
print(f"Corrected gradients: x = {grad_x_corrected:.6f} rad/pixel, y = {grad_y_corrected:.6f} rad/pixel")
x_reduction = abs(grad_x_corrected) / abs(grad_x_original) * 100
y_reduction = abs(grad_y_corrected) / abs(grad_y_original) * 100
print(f"Gradient reduction: x = {x_reduction:.2f}%, y = {y_reduction:.2f}%")

# Verify that amplitude is preserved
amplitude_change = jnp.mean(jnp.abs(amplitude_corrected - amplitude_original))
print(f"\nAmplitude preservation: mean change = {amplitude_change:.8f}")

# Check how well the corrected phase matches the target using unwrapped phases
# Unwrap phases to handle 2π discontinuities properly
phase_corrected_unwrapped = unwrap_phase(phase_corrected)
base_phase_unwrapped = unwrap_phase(base_phase)

# Calculate reconstruction error with unwrapped phases
phase_reconstruction_error = jnp.mean(jnp.abs(phase_corrected_unwrapped - base_phase_unwrapped))
max_reconstruction_error = jnp.max(jnp.abs(phase_corrected_unwrapped - base_phase_unwrapped))
TOLERANCE = 1e-3  # More reasonable tolerance for practical applications

print("\nPhase reconstruction accuracy (unwrapped phases):")
print(f"Mean error from target: {phase_reconstruction_error:.8f} rad")
print(f"Max error from target: {max_reconstruction_error:.8f} rad")
print(f"Reconstruction success: {phase_reconstruction_error < TOLERANCE}")

# Show phase values in the circle region to confirm visibility
center_y, center_x = params.img_center
circle_mask = make_disk(params.img_center, radius, params.img_size_px)

# Use unwrapped phases for more accurate analysis
phase_corrected_unwrapped = unwrap_phase(phase_corrected)
base_phase_unwrapped = unwrap_phase(base_phase)

phase_in_circle_corrected = phase_corrected_unwrapped[circle_mask]
phase_outside_circle_corrected = phase_corrected_unwrapped[~circle_mask]
phase_in_circle_target = base_phase_unwrapped[circle_mask]
phase_outside_circle_target = base_phase_unwrapped[~circle_mask]

print("\nPhase values for visibility (unwrapped):")
print("Corrected phase:")
mean_in = jnp.mean(phase_in_circle_corrected)
std_in = jnp.std(phase_in_circle_corrected)
mean_out = jnp.mean(phase_outside_circle_corrected)
std_out = jnp.std(phase_outside_circle_corrected)
print(f"  Inside circle: {mean_in:.6f} ± {std_in:.6f} rad")
print(f"  Outside circle: {mean_out:.6f} ± {std_out:.6f} rad")
phase_contrast_corrected = jnp.mean(phase_in_circle_corrected) - jnp.mean(phase_outside_circle_corrected)
print(f"  Phase contrast: {phase_contrast_corrected:.6f} rad")

print("Target phase:")
print(f"  Inside circle: {jnp.mean(phase_in_circle_target):.6f} ± {jnp.std(phase_in_circle_target):.6f} rad")
print(f"  Outside circle: {jnp.mean(phase_outside_circle_target):.6f} ± {jnp.std(phase_outside_circle_target):.6f} rad")
print(f"  Phase contrast: {jnp.mean(phase_in_circle_target) - jnp.mean(phase_outside_circle_target):.6f} rad")

# Calculate actual reconstruction accuracy in each region
error_in_circle = jnp.mean(jnp.abs(phase_in_circle_corrected - phase_in_circle_target))
error_outside_circle = jnp.mean(jnp.abs(phase_outside_circle_corrected - phase_outside_circle_target))
CONTRAST_TOLERANCE = 0.1

print("\nRegional reconstruction errors:")
print(f"  Inside circle: {error_in_circle:.6f} rad")
print(f"  Outside circle: {error_outside_circle:.6f} rad")

# Check if phase contrast matches (allowing for sign flip)
contrast_corrected = jnp.mean(phase_in_circle_corrected) - jnp.mean(phase_outside_circle_corrected)
contrast_target = jnp.mean(phase_in_circle_target) - jnp.mean(phase_outside_circle_target)
contrast_match = jnp.abs(jnp.abs(contrast_corrected) - jnp.abs(contrast_target)) < CONTRAST_TOLERANCE
print(f"Phase contrast accuracy: {contrast_match}")
print(f"  |corrected| = {jnp.abs(contrast_corrected):.3f} rad")
print(f"  |target| = {jnp.abs(contrast_target):.3f} rad")

# %%
