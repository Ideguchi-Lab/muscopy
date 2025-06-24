"""
Demonstration of Inline Digital Holography (IDH)
=================================================

This example demonstrates how to use the `muscopy.dh.inline_dh` function to reconstruct
the complex field from an intensity hologram using inline digital holography.
The example shows both basic reconstruction and twin-image suppression techniques.
"""

# %%
# import modules

import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, inline_dh, make_disk, print_all_parameters, propagate_fresnel

# config
SHOW_IMAGE = True

# Set matplotlib backend for compatibility
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
except ImportError:
    pass

print(f"JAX backend: {jax.default_backend()}")

# %%
# set Microscopy parameters

params = MuParameters(
    na=0.1,  # Lower NA for inline holography to avoid aliasing
    wavelength_m=532e-9,  # Green laser wavelength
    img_size_px=512,
    px_size_m=6.5e-6,  # Typical camera pixel size
    n_sol=1.0,  # Air medium
)

# show parameters
print_all_parameters(params, show_properties=True)

# %%
# Create a synthetic object for demonstration

# Object distance
z_obj_m = 30e-3  # 30 mm from sensor

# Create a simple object: two disks with different phases
radius1 = 40
radius2 = 20
pos1 = (params.img_center[0] - 50, params.img_center[1] - 30)
pos2 = (params.img_center[0] + 40, params.img_center[1] + 20)

# Create object transmission function
obj_disk1 = make_disk(pos1, radius1, params.img_size_px)
obj_disk2 = make_disk(pos2, radius2, params.img_size_px)

# Object with phase shifts
phase_shift1 = math.pi / 2  # 90 degrees
phase_shift2 = math.pi  # 180 degrees
obj_transmission = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.complex64)
obj_transmission = jnp.where(obj_disk1, jnp.exp(1j * phase_shift1), obj_transmission)
obj_transmission = jnp.where(obj_disk2, jnp.exp(1j * phase_shift2), obj_transmission)

# Apply aperture limitation
aperture = make_disk(params.img_center, params.img_size_px // 4, params.img_size_px)
obj_transmission = jnp.where(aperture, obj_transmission, 1.0)

# %%
# Simulate hologram formation

# Propagate object field to sensor plane
obj_field_at_sensor = propagate_fresnel(obj_transmission, params, z_obj_m)

# Create hologram (intensity pattern)
hologram = jnp.abs(obj_field_at_sensor) ** 2

# Add some noise to make it more realistic
key = jax.random.PRNGKey(42)
noise_level = 0.01
noise = jax.random.normal(key, hologram.shape) * noise_level * jnp.mean(hologram)
hologram = hologram + noise
hologram = jnp.clip(hologram, 0, None)  # Ensure non-negative

if SHOW_IMAGE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original object phase
    obj_phase = jax.device_get(jnp.angle(obj_transmission))
    im1 = axes[0].imshow(obj_phase, cmap="hsv", vmin=-math.pi, vmax=math.pi)
    axes[0].set_title("Original Object Phase")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Object amplitude
    obj_amp = jax.device_get(jnp.abs(obj_transmission))
    im2 = axes[1].imshow(obj_amp, cmap="gray")
    axes[1].set_title("Original Object Amplitude")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Hologram
    hologram_show = jax.device_get(hologram)
    im3 = axes[2].imshow(hologram_show, cmap="gray")
    axes[2].set_title("Hologram (Intensity)")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

# %%
# Reconstruct using inline digital holography

print("Reconstructing with inline DH...")

# Basic reconstruction without twin-image suppression
reconstructed_basic = inline_dh(hologram, params, z_obj_m, twin_iter=0)

# Reconstruction with twin-image suppression
reconstructed_twin_suppressed = inline_dh(hologram, params, z_obj_m, twin_iter=5)

# %%
# Compare results

if SHOW_IMAGE:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Basic reconstruction
    phase_basic = jax.device_get(jnp.angle(reconstructed_basic))
    amp_basic = jax.device_get(jnp.abs(reconstructed_basic))

    im1 = axes[0, 0].imshow(phase_basic, cmap="hsv", vmin=-math.pi, vmax=math.pi)
    axes[0, 0].set_title("Basic IDH - Phase")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(amp_basic, cmap="gray")
    axes[0, 1].set_title("Basic IDH - Amplitude")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Show phase difference from original
    phase_diff_basic = jax.device_get(jnp.angle(obj_transmission) - jnp.angle(reconstructed_basic))
    im3 = axes[0, 2].imshow(phase_diff_basic, cmap="RdBu", vmin=-math.pi, vmax=math.pi)
    axes[0, 2].set_title("Phase Difference (Original - Basic)")
    axes[0, 2].axis("off")
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Twin-image suppressed reconstruction
    phase_twin = jax.device_get(jnp.angle(reconstructed_twin_suppressed))
    amp_twin = jax.device_get(jnp.abs(reconstructed_twin_suppressed))

    im4 = axes[1, 0].imshow(phase_twin, cmap="hsv", vmin=-math.pi, vmax=math.pi)
    axes[1, 0].set_title("Twin-Suppressed IDH - Phase")
    axes[1, 0].axis("off")
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    im5 = axes[1, 1].imshow(amp_twin, cmap="gray")
    axes[1, 1].set_title("Twin-Suppressed IDH - Amplitude")
    axes[1, 1].axis("off")
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    # Show phase difference from original
    phase_diff_twin = jax.device_get(jnp.angle(obj_transmission) - jnp.angle(reconstructed_twin_suppressed))
    im6 = axes[1, 2].imshow(phase_diff_twin, cmap="RdBu", vmin=-math.pi, vmax=math.pi)
    axes[1, 2].set_title("Phase Difference (Original - Twin-Suppressed)")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.show()

# %%
# Analyze reconstruction quality

# Calculate RMS error for central region
center_crop_size = 200
center_start = (params.img_size_px - center_crop_size) // 2
center_end = center_start + center_crop_size

original_center = obj_transmission[center_start:center_end, center_start:center_end]
basic_center = reconstructed_basic[center_start:center_end, center_start:center_end]
twin_center = reconstructed_twin_suppressed[center_start:center_end, center_start:center_end]

# Phase RMS error
phase_rms_basic = jnp.sqrt(jnp.mean((jnp.angle(original_center) - jnp.angle(basic_center)) ** 2))
phase_rms_twin = jnp.sqrt(jnp.mean((jnp.angle(original_center) - jnp.angle(twin_center)) ** 2))

print("\n=== Reconstruction Quality Analysis ===")
print(f"Object distance: {z_obj_m * 1000:.1f} mm")
print(f"Phase RMS error (Basic IDH): {phase_rms_basic:.4f} rad")
print(f"Phase RMS error (Twin-suppressed IDH): {phase_rms_twin:.4f} rad")
print(f"Improvement factor: {phase_rms_basic / phase_rms_twin:.2f}x")

# %%
# Demonstrate distance dependency

if SHOW_IMAGE:
    print("\nDemonstrating reconstruction at different distances...")

    distances = [20e-3, 30e-3, 40e-3, 50e-3]  # 20, 30, 40, 50 mm

    fig, axes = plt.subplots(1, len(distances), figsize=(20, 5))

    for i, dist in enumerate(distances):
        reconstructed_dist = inline_dh(hologram, params, dist, twin_iter=3)
        phase_dist = jax.device_get(jnp.angle(reconstructed_dist))

        im = axes[i].imshow(phase_dist, cmap="hsv", vmin=-math.pi, vmax=math.pi)
        axes[i].set_title(f"Distance: {dist * 1000:.0f} mm")
        axes[i].axis("off")
        plt.colorbar(im, ax=axes[i], fraction=0.046)

    plt.suptitle("IDH Reconstruction at Different Object Distances")
    plt.tight_layout()
    plt.show()

print("\n=== IDH Demonstration Complete ===")
print("This example showed:")
print("1. Synthetic hologram generation")
print("2. Basic inline DH reconstruction")
print("3. Twin-image suppression")
print("4. Quality comparison")
print("5. Distance dependency")
