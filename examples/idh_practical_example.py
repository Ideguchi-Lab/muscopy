"""
Practical Inline Digital Holography Example
===========================================

This example demonstrates a more practical use case of inline digital holography
for analyzing a biological sample. It shows how to handle real-world scenarios
including noise, background subtraction, and parameter optimization.
"""

# %%
# import modules


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from muscopy.dh import MuParameters, inline_dh, make_disk, print_all_parameters, propagate_fresnel

# Set matplotlib backend for compatibility
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
except ImportError:
    pass

# config
SHOW_IMAGE = True
SAVE_RESULTS = False

print(f"JAX backend: {jax.default_backend()}")

# %%
# Set up experimental parameters for biological imaging

params = MuParameters(
    na=0.2,  # Moderate NA for good resolution while avoiding aliasing
    wavelength_m=635e-9,  # Red laser wavelength (common in bio-imaging)
    img_size_px=1024,  # Higher resolution for detailed imaging
    px_size_m=5.5e-6,  # Typical scientific camera pixel size
    n_sol=1.33,  # Water medium (biological samples)
)

print_all_parameters(params, show_properties=True)

# %%
# Simulate a biological cell

# Object distance (typical working distance for microscopy)
z_obj_m = 25e-3  # 25 mm


# Create a cell-like object with organelles
def create_cell_sample(params):
    """Create a synthetic cell with various organelles."""
    # Cell body (circular)
    cell_radius = 80
    cell_center = params.img_center
    cell_mask = make_disk(cell_center, cell_radius, params.img_size_px)

    # Nucleus (smaller circle with higher phase)
    nucleus_radius = 25
    nucleus_center = (cell_center[0] - 10, cell_center[1] + 5)
    nucleus_mask = make_disk(nucleus_center, nucleus_radius, params.img_size_px)

    # Organelles (small high-phase regions)
    organelles = []
    organelle_positions = [
        (cell_center[0] + 30, cell_center[1] - 20),
        (cell_center[0] - 25, cell_center[1] - 30),
        (cell_center[0] + 15, cell_center[1] + 35),
        (cell_center[0] - 40, cell_center[1] + 10),
    ]

    for pos in organelle_positions:
        organelle_mask = make_disk(pos, 8, params.img_size_px)
        organelles.append(organelle_mask)

    # Create transmission function
    transmission = jnp.ones((params.img_size_px, params.img_size_px), dtype=jnp.complex64)

    # Cell body: slight phase shift and amplitude reduction
    cell_phase = 0.3  # Small phase shift typical of cells
    cell_amplitude = 0.95  # Slight absorption
    transmission = jnp.where(cell_mask, cell_amplitude * jnp.exp(1j * cell_phase), transmission)

    # Nucleus: higher phase shift and more absorption
    nucleus_phase = 0.8
    nucleus_amplitude = 0.85
    transmission = jnp.where(nucleus_mask, nucleus_amplitude * jnp.exp(1j * nucleus_phase), transmission)

    # Organelles: high phase, low absorption
    organelle_phase = 1.2
    organelle_amplitude = 0.9
    for organelle_mask in organelles:
        transmission = jnp.where(organelle_mask, organelle_amplitude * jnp.exp(1j * organelle_phase), transmission)

    return transmission


# Create the sample
sample_transmission = create_cell_sample(params)

# %%
# Simulate realistic hologram formation with noise and background


def simulate_realistic_hologram(transmission, params, z_obj_m, noise_level=0.05):
    """Simulate a realistic hologram with noise and background."""
    # Propagate to sensor
    field_at_sensor = propagate_fresnel(transmission, params, z_obj_m)

    # Add uniform background (illumination inhomogeneity)
    xx, yy = jnp.meshgrid(jnp.arange(params.img_size_px), jnp.arange(params.img_size_px), indexing="ij")
    background_pattern = 1.0 + 0.1 * jnp.sin(2 * jnp.pi * xx / params.img_size_px) * jnp.cos(
        2 * jnp.pi * yy / params.img_size_px
    )

    # Create hologram
    hologram = jnp.abs(field_at_sensor) ** 2 * background_pattern

    # Add shot noise
    key = jax.random.PRNGKey(123)
    noise = jax.random.poisson(key, hologram / noise_level) * noise_level - hologram
    hologram = hologram + noise

    # Add dark current and offset
    dark_current = 50.0  # typical camera dark current in counts
    hologram = hologram + dark_current

    # Quantization (simulate A/D conversion)
    hologram = jnp.round(hologram).astype(jnp.float32)
    hologram = jnp.clip(hologram, 0, 65535)  # 16-bit camera

    return hologram, background_pattern


hologram_raw, background = simulate_realistic_hologram(sample_transmission, params, z_obj_m)

# %%
# Preprocessing: background subtraction and normalization

# Simple background estimation (could be measured experimentally)
background_estimated = jnp.mean(hologram_raw) * background
hologram_corrected = hologram_raw - background_estimated + jnp.mean(background_estimated)

# Normalize to [0, 1] range
hologram_normalized = (hologram_corrected - jnp.min(hologram_corrected)) / (
    jnp.max(hologram_corrected) - jnp.min(hologram_corrected)
)

if SHOW_IMAGE:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original sample
    sample_phase = jax.device_get(jnp.angle(sample_transmission))
    im1 = axes[0].imshow(sample_phase, cmap="viridis", vmin=0, vmax=1.5)
    axes[0].set_title("Original Sample Phase")
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Raw hologram
    hologram_raw_show = jax.device_get(hologram_raw)
    im2 = axes[1].imshow(hologram_raw_show, cmap="gray")
    axes[1].set_title("Raw Hologram")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Corrected hologram
    hologram_corrected_show = jax.device_get(hologram_normalized)
    im3 = axes[2].imshow(hologram_corrected_show, cmap="gray")
    axes[2].set_title("Preprocessed Hologram")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.show()

# %%
# IDH reconstruction with different parameters


def analyze_reconstruction_quality(hologram, original, params, z_obj_m, twin_iterations):
    """Analyze reconstruction quality for different twin suppression iterations."""
    results = {}

    for twin_iter in twin_iterations:
        reconstructed = inline_dh(hologram, params, z_obj_m, twin_iter=twin_iter)

        # Calculate metrics in central region
        center_size = 200
        center_start = (params.img_size_px - center_size) // 2
        center_end = center_start + center_size

        orig_center = original[center_start:center_end, center_start:center_end]
        recon_center = reconstructed[center_start:center_end, center_start:center_end]

        # Phase correlation
        phase_orig = jnp.angle(orig_center)
        phase_recon = jnp.angle(recon_center)
        phase_corr = jnp.corrcoef(phase_orig.flatten(), phase_recon.flatten())[0, 1]

        # RMS error
        phase_rms = jnp.sqrt(jnp.mean((phase_orig - phase_recon) ** 2))

        results[twin_iter] = {"reconstructed": reconstructed, "phase_correlation": phase_corr, "phase_rms": phase_rms}

    return results


# Test different twin suppression parameters
twin_iterations = [0, 2, 5, 10]
print("\\nAnalyzing reconstruction quality...")

results = analyze_reconstruction_quality(hologram_normalized, sample_transmission, params, z_obj_m, twin_iterations)

# Print quality metrics
print("\\n=== Reconstruction Quality Analysis ===")
for twin_iter in twin_iterations:
    result = results[twin_iter]
    print(
        f"Twin iterations: {twin_iter:2d} | "
        f"Phase correlation: {result['phase_correlation']:.3f} | "
        f"Phase RMS error: {result['phase_rms']:.4f} rad"
    )

# %%
# Visualize results

if SHOW_IMAGE:
    fig, axes = plt.subplots(2, len(twin_iterations), figsize=(20, 10))

    for i, twin_iter in enumerate(twin_iterations):
        reconstructed = results[twin_iter]["reconstructed"]

        # Phase reconstruction
        phase_recon = jax.device_get(jnp.angle(reconstructed))
        im1 = axes[0, i].imshow(phase_recon, cmap="viridis", vmin=0, vmax=1.5)
        axes[0, i].set_title(f"Phase (Twin iter: {twin_iter})")
        axes[0, i].axis("off")
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

        # Amplitude reconstruction
        amp_recon = jax.device_get(jnp.abs(reconstructed))
        im2 = axes[1, i].imshow(amp_recon, cmap="gray", vmin=0.8, vmax=1.0)
        axes[1, i].set_title(f"Amplitude (Twin iter: {twin_iter})")
        axes[1, i].axis("off")
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

    plt.tight_layout()
    plt.show()

# %%
# Distance optimization


def find_optimal_distance(hologram, params, distance_range, twin_iter=5):
    """Find optimal reconstruction distance by maximizing focus metric."""
    distances = jnp.linspace(distance_range[0], distance_range[1], 20)
    focus_metrics = []

    print("\\nOptimizing reconstruction distance...")

    for i, dist in enumerate(distances):
        reconstructed = inline_dh(hologram, params, dist, twin_iter=twin_iter)

        # Focus metric: variance of phase gradient (higher = more focused)
        phase = jnp.angle(reconstructed)
        grad_x = jnp.diff(phase, axis=1)
        grad_y = jnp.diff(phase, axis=0)
        focus_metric = jnp.var(grad_x) + jnp.var(grad_y)
        focus_metrics.append(focus_metric)

        if i % 5 == 0:
            print(f"  Distance: {dist * 1000:.1f} mm, Focus metric: {focus_metric:.6f}")

    focus_metrics = jnp.array(focus_metrics)
    optimal_idx = jnp.argmax(focus_metrics)
    optimal_distance = distances[optimal_idx]

    return distances, focus_metrics, optimal_distance


# Optimize distance around the true value
distance_range = (20e-3, 35e-3)  # Search around 25 mm
distances, focus_metrics, optimal_distance = find_optimal_distance(hologram_normalized, params, distance_range)

print(f"\\nOptimal distance: {optimal_distance * 1000:.2f} mm (true: {z_obj_m * 1000:.1f} mm)")

# %%
# Final reconstruction with optimized parameters

print("\\nPerforming final reconstruction...")
final_reconstruction = inline_dh(hologram_normalized, params, optimal_distance, twin_iter=5)

if SHOW_IMAGE:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Original sample
    original_phase = jax.device_get(jnp.angle(sample_transmission))
    original_amp = jax.device_get(jnp.abs(sample_transmission))

    im1 = axes[0, 0].imshow(original_phase, cmap="viridis", vmin=0, vmax=1.5)
    axes[0, 0].set_title("Original Phase")
    axes[0, 0].axis("off")
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[1, 0].imshow(original_amp, cmap="gray", vmin=0.8, vmax=1.0)
    axes[1, 0].set_title("Original Amplitude")
    axes[1, 0].axis("off")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    # Reconstructed
    recon_phase = jax.device_get(jnp.angle(final_reconstruction))
    recon_amp = jax.device_get(jnp.abs(final_reconstruction))

    im3 = axes[0, 1].imshow(recon_phase, cmap="viridis", vmin=0, vmax=1.5)
    axes[0, 1].set_title("Reconstructed Phase")
    axes[0, 1].axis("off")
    plt.colorbar(im3, ax=axes[0, 1], fraction=0.046)

    im4 = axes[1, 1].imshow(recon_amp, cmap="gray", vmin=0.8, vmax=1.0)
    axes[1, 1].set_title("Reconstructed Amplitude")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Distance optimization curve
    distances_mm = jax.device_get(distances) * 1000
    focus_metrics_show = jax.device_get(focus_metrics)
    axes[0, 2].plot(distances_mm, focus_metrics_show, "b-", linewidth=2)
    axes[0, 2].axvline(
        optimal_distance * 1000, color="r", linestyle="--", label=f"Optimal: {optimal_distance * 1000:.1f} mm"
    )
    axes[0, 2].axvline(z_obj_m * 1000, color="g", linestyle="--", label=f"True: {z_obj_m * 1000:.1f} mm")
    axes[0, 2].set_xlabel("Distance (mm)")
    axes[0, 2].set_ylabel("Focus Metric")
    axes[0, 2].set_title("Distance Optimization")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Phase difference
    phase_diff = original_phase - recon_phase
    im6 = axes[1, 2].imshow(phase_diff, cmap="RdBu", vmin=-0.5, vmax=0.5)
    axes[1, 2].set_title("Phase Difference")
    axes[1, 2].axis("off")
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    plt.show()

# %%
# Final analysis and summary

# Calculate final quality metrics
center_size = 300
center_start = (params.img_size_px - center_size) // 2
center_end = center_start + center_size

orig_center = sample_transmission[center_start:center_end, center_start:center_end]
final_center = final_reconstruction[center_start:center_end, center_start:center_end]

final_phase_corr = jnp.corrcoef(jnp.angle(orig_center).flatten(), jnp.angle(final_center).flatten())[0, 1]

final_phase_rms = jnp.sqrt(jnp.mean((jnp.angle(orig_center) - jnp.angle(final_center)) ** 2))

print("\\n" + "=" * 50)
print("PRACTICAL IDH RECONSTRUCTION SUMMARY")
print("=" * 50)
print("Sample type: Synthetic biological cell")
print(f"Wavelength: {params.wavelength_m * 1e9:.0f} nm")
print(f"Pixel size: {params.px_size_m * 1e6:.1f} μm")
print(f"Image size: {params.img_size_px} × {params.img_size_px} pixels")
print(f"Medium: n = {params.n_sol}")
print(f"True object distance: {z_obj_m * 1000:.1f} mm")
print(f"Optimized distance: {optimal_distance * 1000:.2f} mm")
print(f"Distance error: {abs(optimal_distance - z_obj_m) * 1000:.2f} mm")
print(f"Final phase correlation: {final_phase_corr:.3f}")
print(f"Final phase RMS error: {final_phase_rms:.4f} rad")
print("\\nKey findings:")
print("- Twin-image suppression improves reconstruction quality")
print("- Distance optimization is crucial for accurate results")
print("- Preprocessing (background subtraction) is important")
print("- IDH can resolve cellular structures effectively")

if SAVE_RESULTS:
    print("\\nSaving results...")
    jnp.save("hologram_corrected.npy", hologram_normalized)
    jnp.save("idh_reconstruction.npy", final_reconstruction)
    print("Results saved to .npy files")

print("\\nPractical IDH demonstration complete!")
