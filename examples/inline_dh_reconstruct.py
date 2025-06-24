"""
Demonstration of Inline Digital Holography (IDH) reconstruction
===============================================================

This example demonstrates how to use the `muscopy.dh.inline_dh` function to reconstruct
complex amplitude from phase-shifted inline holograms. The example shows the workflow
for Phase-Shifting Inline Digital Holography (PS-IDH) with synthetic data.

Unlike off-axis digital holography, inline DH can overcome sampling resolution limitations
imposed by the Nyquist-Shannon limit for large phase shifts.
"""

# %%
# import modules

import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, inline_dh, make_disk, print_all_parameters

# config
SHOW_IMAGE = True

print(f"JAX backend: {jax.default_backend()}")

# %%
# Set microscopy parameters

params = MuParameters(
    na=0.6,
    wavelength_m=633e-9,  # HeNe laser
    img_size_px=256,
    px_size_m=5.2e-6,  # Camera pixel size
    n_sol=1.0,  # Air medium
)

# Show microscopy parameters
print_all_parameters(params, show_properties=True)

# %%
# Create synthetic object field
print("Creating synthetic object field...")

# Create a phase object (circular disk with phase step)
center = params.img_center
radius = 40
phase_disk = make_disk(center, radius, params.img_size_px)

# Create amplitude modulation (Gaussian)
xx, yy = jnp.meshgrid(jnp.arange(params.img_size_px), jnp.arange(params.img_size_px), indexing="ij")
gaussian_amp = jnp.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * (radius / 2)**2))

# Combined object field: amplitude modulation + phase step
phase_step = math.pi / 2  # 90 degree phase step
object_field = gaussian_amp * jnp.exp(1j * phase_step * phase_disk)

print(f"Object field shape: {object_field.shape}")
print(f"Object field amplitude range: {jnp.min(jnp.abs(object_field)):.3f} - {jnp.max(jnp.abs(object_field)):.3f}")
print(f"Object field phase range: {jnp.min(jnp.angle(object_field)):.3f} - {jnp.max(jnp.angle(object_field)):.3f} rad")

# %%
# Generate phase-shifted holograms
print("\\nGenerating phase-shifted holograms...")

# Define phase shifts for 4-step phase shifting
num_steps = 4
phase_shifts = jnp.linspace(0, 2 * math.pi, num_steps, endpoint=False)
ref_amplitude = 1.0

print(f"Phase shifts: {phase_shifts} rad")
print(f"Phase shifts (degrees): {jnp.degrees(phase_shifts)}")

# Generate hologram intensity stack
holograms = jnp.zeros((num_steps, params.img_size_px, params.img_size_px))

for i, delta in enumerate(phase_shifts):
    # Hologram field: object + reference wave with phase shift
    reference_field = ref_amplitude * jnp.exp(1j * delta)
    hologram_field = object_field + reference_field

    # Intensity measurement
    intensity = jnp.abs(hologram_field)**2
    holograms = holograms.at[i].set(intensity)

print(f"Hologram stack shape: {holograms.shape}")
print(f"Intensity range: {jnp.min(holograms):.3f} - {jnp.max(holograms):.3f}")

# %%
# Reconstruct using Phase-Shifting Inline Digital Holography
print("\\nPerforming PS-IDH reconstruction...")

reconstructed_field = inline_dh(
    i_stack=holograms,
    deltas=phase_shifts,
    params=params,
    ref_amp=ref_amplitude
)

print(f"Reconstructed field shape: {reconstructed_field.shape}")
print(
    f"Reconstructed amplitude range: {jnp.min(jnp.abs(reconstructed_field)):.3f} - "
    f"{jnp.max(jnp.abs(reconstructed_field)):.3f}"
)
print(
    f"Reconstructed phase range: {jnp.min(jnp.angle(reconstructed_field)):.3f} - "
    f"{jnp.max(jnp.angle(reconstructed_field)):.3f} rad"
)

# %%
# Calculate reconstruction quality metrics
print("\\nCalculating reconstruction quality...")

# Amplitude correlation
original_amp = jnp.abs(object_field)
reconstructed_amp = jnp.abs(reconstructed_field)
amp_correlation = jnp.corrcoef(original_amp.flatten(), reconstructed_amp.flatten())[0, 1]

# Phase correlation (only in object region)
original_phase = jnp.angle(object_field)
reconstructed_phase = jnp.angle(reconstructed_field)
# Mask for object region (where amplitude > 10% of maximum)
object_mask = original_amp > 0.1 * jnp.max(original_amp)
# Unwrap phase difference for better comparison
phase_diff = jnp.angle(jnp.exp(1j * (original_phase - reconstructed_phase)))
phase_rmse = jnp.sqrt(jnp.mean((phase_diff[object_mask])**2))

print(f"Amplitude correlation: {amp_correlation:.4f}")
print(f"Phase RMSE in object region: {phase_rmse:.4f} rad ({jnp.degrees(phase_rmse):.2f} degrees)")

# %%
# Visualization
if SHOW_IMAGE:
    print("\\nGenerating visualization...")

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))

    # Row 1: Input holograms
    for i in range(num_steps):
        ax = axes[0, i]
        im = ax.imshow(holograms[i], cmap='gray')
        ax.set_title(f'Hologram {i + 1}\\nδ = {jnp.degrees(phase_shifts[i]):.0f}°')
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.6)

    # Row 2: Original object
    ax = axes[1, 0]
    im = ax.imshow(jnp.abs(object_field), cmap='viridis')
    ax.set_title('Original Amplitude')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes[1, 1]
    im = ax.imshow(jnp.angle(object_field), cmap='hsv', vmin=-math.pi, vmax=math.pi)
    ax.set_title('Original Phase')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Row 3: Reconstructed object
    ax = axes[1, 2]
    im = ax.imshow(jnp.abs(reconstructed_field), cmap='viridis')
    ax.set_title('Reconstructed Amplitude')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax = axes[1, 3]
    im = ax.imshow(jnp.angle(reconstructed_field), cmap='hsv', vmin=-math.pi, vmax=math.pi)
    ax.set_title('Reconstructed Phase')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Row 3: Difference plots
    amp_diff = jnp.abs(reconstructed_field) - jnp.abs(object_field)
    ax = axes[2, 0]
    im = ax.imshow(amp_diff, cmap='RdBu_r')
    ax.set_title('Amplitude Difference')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    phase_diff_wrapped = jnp.angle(jnp.exp(1j * (jnp.angle(reconstructed_field) - jnp.angle(object_field))))
    ax = axes[2, 1]
    im = ax.imshow(phase_diff_wrapped, cmap='RdBu_r', vmin=-math.pi, vmax=math.pi)
    ax.set_title('Phase Difference (wrapped)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, shrink=0.6)

    # Line profiles
    center_line = params.img_center[0]
    ax = axes[2, 2]
    ax.plot(jnp.abs(object_field[center_line, :]), 'b-', label='Original', linewidth=2)
    ax.plot(jnp.abs(reconstructed_field[center_line, :]), 'r--', label='Reconstructed', linewidth=2)
    ax.set_title('Amplitude Profile')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2, 3]
    ax.plot(jnp.angle(object_field[center_line, :]), 'b-', label='Original', linewidth=2)
    ax.plot(jnp.angle(reconstructed_field[center_line, :]), 'r--', label='Reconstructed', linewidth=2)
    ax.set_title('Phase Profile')
    ax.set_xlabel('Pixel')
    ax.set_ylabel('Phase (rad)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle('Phase-Shifting Inline Digital Holography Reconstruction', fontsize=16, y=0.98)
    plt.show()

# %%
# Summary
print("\\n" + "=" * 60)
print("PHASE-SHIFTING INLINE DIGITAL HOLOGRAPHY RECONSTRUCTION")
print("=" * 60)
print(f"Number of phase shifts: {num_steps}")
print(f"Phase shift step: {360 / num_steps:.1f} degrees")
print(f"Reference amplitude: {ref_amplitude}")
print("Reconstruction quality:")
print(f"  - Amplitude correlation: {amp_correlation:.4f}")
print(f"  - Phase RMSE: {phase_rmse:.4f} rad ({jnp.degrees(phase_rmse):.2f}°)")
print("\\nPS-IDH provides high-quality reconstruction without off-axis sampling limitations!")
