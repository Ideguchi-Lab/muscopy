"""
IDT with MLB Simulation Example
===============================

This example demonstrates how to use Intensity Diffraction Tomography (IDT)
with Multi-layer Born (MLB) simulation for microscopy analysis.

The private muscopy_mlbsim package must be installed separately.
See :ref:`mlb-installation` for repository access and installation instructions.
Without the simulator, this example skips execution.
"""

import gc
import warnings
from collections.abc import Sequence
from importlib.util import find_spec

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from muscopy.example_mlb import simulate_mlb_measurements
from muscopy.idt import IDTConfig, IDTParameters, compute_g_list, compute_idt, transfer_func_im, transfer_func_re

# Suppress JAX warnings about dtype conversion that can interfere with execution
warnings.filterwarnings("ignore", category=FutureWarning, message=".*scatter inputs have incompatible types.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")

_DIAGNOSTIC_EPSILON = 1e-6


def _robust_symmetric_limit(array: np.ndarray, *, expected_scale: float) -> float:
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return expected_scale
    percentile_limit = float(np.percentile(np.abs(finite_values), 99.5))
    return max(percentile_limit, expected_scale)


def _robust_image_limits(array: np.ndarray) -> tuple[float, float]:
    finite_values = array[np.isfinite(array)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite_values, [1.0, 99.0])
    if np.isclose(vmin, vmax):
        return float(np.min(finite_values)), float(np.max(finite_values))
    return float(vmin), float(vmax)


def _visualize_results(  # noqa: PLR0914, PLR0915
    n_reconstructed: Array,
    target_intensity_images: Sequence[Array],
    delta_n: float,
) -> None:
    """Create visualizations of the reconstruction results."""
    print("Creating visualizations...")

    # Convert to numpy for matplotlib
    n_reconstructed_np = np.asarray(n_reconstructed)

    # Cross-sections through the center
    center_x = n_reconstructed_np.shape[0] // 2
    center_y = n_reconstructed_np.shape[1] // 2
    center_z = n_reconstructed_np.shape[2] // 2

    # Create figure with subplots
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original hologram
    hol_display = np.asarray(target_intensity_images[0])

    image_vmin, image_vmax = _robust_image_limits(hol_display)
    im1 = axes[0, 0].imshow(hol_display, cmap="gray", vmin=image_vmin, vmax=image_vmax)
    axes[0, 0].set_title("Sample Intensity")
    axes[0, 0].set_xlabel("x [px]")
    axes[0, 0].set_ylabel("y [px]")
    plt.colorbar(im1, ax=axes[0, 0])

    # Cross-sections of reconstruction
    vmax = _robust_symmetric_limit(n_reconstructed_np, expected_scale=delta_n)
    vmin = -vmax
    im2 = axes[0, 1].imshow(n_reconstructed_np[:, :, center_z], cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("XY Cross-section (Center Z)")
    axes[0, 1].set_xlabel("x [px]")
    axes[0, 1].set_ylabel("y [px]")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(n_reconstructed_np[:, center_y, :], cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title("XZ Cross-section (Center Y)")
    axes[0, 2].set_xlabel("z [px]")
    axes[0, 2].set_ylabel("x [px]")
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(n_reconstructed_np[center_x, :, :], cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("YZ Cross-section (Center X)")
    axes[1, 0].set_xlabel("z [px]")
    axes[1, 0].set_ylabel("y [px]")
    plt.colorbar(im4, ax=axes[1, 0])

    # Profile through center
    profile = n_reconstructed_np[center_x, center_y, :]
    axes[1, 1].plot(profile)
    axes[1, 1].set_title("Central Profile (Z direction)")
    axes[1, 1].set_xlabel("z [px]")
    axes[1, 1].set_ylabel("Δn")
    axes[1, 1].set_ylim(vmin, vmax)
    axes[1, 1].grid(True)

    # Show max absolute projection
    max_proj = np.max(np.abs(n_reconstructed_np), axis=2)
    im6 = axes[1, 2].imshow(max_proj, cmap="magma", vmin=0, vmax=vmax)
    axes[1, 2].set_title("Max |Δn| Projection (Z axis)")
    axes[1, 2].set_xlabel("x [px]")
    axes[1, 2].set_ylabel("y [px]")
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("idt_mlb_simulation_results.png", dpi=150, bbox_inches="tight")
    print("IDT reconstruction visualization saved to idt_mlb_simulation_results.png")
    plt.close()  # Close the figure to free memory

    # Print summary
    print("\nReconstruction Summary:")
    print(f"Shape: {n_reconstructed_np.shape}")
    print(f"Δn range: [{n_reconstructed_np.min():.4f}, {n_reconstructed_np.max():.4f}]")
    print(f"Robust |Δn| display limit: {vmax:.4f}")
    print(f"Expected Δn: {delta_n:.4f}")
    print(f"Peak |Δn|: {np.max(np.abs(n_reconstructed_np)):.4f}")
    print(f"Peak |Δn| / expected Δn: {np.max(np.abs(n_reconstructed_np)) / delta_n:.2f}")


def _print_idt_input_diagnostics(  # noqa: PLR0914
    idt_params: IDTParameters,
    target_intensity_images: Sequence[Array],
    ref_intensity_images: Sequence[Array],
    u_illumination_list: Sequence[tuple[float, float]],
    config: IDTConfig,
) -> None:
    ref_values = np.concatenate([np.asarray(ref).ravel() for ref in ref_intensity_images])
    g_values = np.concatenate(
        [
            np.asarray(g).ravel()
            for g in compute_g_list(
                target_intensity_images,
                ref_intensity_images,
                precision=config.precision,
                ref_floor_ratio=config.ref_floor_ratio,
                g_clip=config.g_clip,
            )
        ]
    )

    h_re = []
    h_im = []
    for u_illumination, ref_image in zip(u_illumination_list, ref_intensity_images, strict=True):
        incident_intensity = float(jnp.mean(ref_image))
        h_re.append(
            transfer_func_re(
                idt_params,
                u_illumination,
                z=0.0,
                incident_intensity=incident_intensity,
                precision=config.precision,
            )
            / incident_intensity
        )
        h_im.append(
            transfer_func_im(
                idt_params,
                u_illumination,
                z=0.0,
                incident_intensity=incident_intensity,
                precision=config.precision,
            )
            / incident_intensity
        )

    h_re_stack = jnp.stack(h_re, axis=-1)
    h_im_stack = jnp.stack(h_im, axis=-1)
    h_norm_scale = jnp.maximum(jnp.max(jnp.abs(h_re_stack)), jnp.max(jnp.abs(h_im_stack)))
    h_norm_scale = jnp.where(h_norm_scale < _DIAGNOSTIC_EPSILON, 1.0, h_norm_scale)
    h_re_scaled = h_re_stack / h_norm_scale
    h_im_scaled = h_im_stack / h_norm_scale
    sum_h_re = jnp.sum(jnp.abs(h_re_scaled) ** 2, axis=-1)
    sum_h_im = jnp.sum(jnp.abs(h_im_scaled) ** 2, axis=-1)
    term1 = sum_h_re * sum_h_im
    term2 = jnp.abs(jnp.sum(jnp.conjugate(h_re_scaled) * h_im_scaled, axis=-1)) ** 2
    det = np.asarray(jnp.maximum(jnp.real(term1 - term2), 0.0))
    support = np.asarray((sum_h_re + sum_h_im) > _DIAGNOSTIC_EPSILON)
    det_support = det[support]

    print("\nIDT input diagnostics:")
    print(
        "Reference intensity: "
        f"min={np.min(ref_values):.6g}, p1={np.percentile(ref_values, 1):.6g}, "
        f"mean={np.mean(ref_values):.6g}, p99={np.percentile(ref_values, 99):.6g}"
    )
    print(
        "Normalized contrast |g|: "
        f"max={np.max(np.abs(g_values)):.6g}, p99.9={np.percentile(np.abs(g_values), 99.9):.6g}"
    )
    print(f"Transfer h_norm_scale at z=0: {float(h_norm_scale):.6g}")
    if det_support.size > 0:
        print(
            "Coupled inverse determinant at z=0: "
            f"min={np.min(det_support):.6g}, p1={np.percentile(det_support, 1):.6g}, "
            f"median={np.median(det_support):.6g}"
        )
    else:
        print("Coupled inverse determinant at z=0: no supported frequencies")


# %%
# Main execution
def main() -> None:
    """Demonstrate IDT with MLB simulation."""
    if find_spec("muscopy_mlbsim") is None:
        print(
            "Skipping IDT demo because muscopy-mlbsim is unavailable. See "
            "https://muscopy.readthedocs.io/en/latest/mlb_installation.html for installation instructions."
        )
        return

    # Clear JAX compilation cache at the start to prevent memory accumulation
    jax.clear_caches()  # type: ignore[no-untyped-call]

    idt_params = IDTParameters(
        na=0.8,
        wavelength_m=532e-9,
        img_size_px=1023,
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.6,
        num_z_slices=64,
    )
    delta_n = 0.01
    target_intensity_images, ref_intensity_images, u_illumination_list = simulate_mlb_measurements(
        idt_params, radius_um=3.0, delta_n=delta_n, num_angles=10
    )

    # Clear any cached data before IDT computation
    print("Clearing JAX cache before IDT computation...")
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()

    print("Starting IDT computation...")
    print(f"IDT parameters: img_size_px={idt_params.img_size_px}, num_z_slices={idt_params.num_z_slices}")
    print(f"Number of illumination angles: {len(u_illumination_list)}")
    print(f"Memory usage before IDT: {sum(img.nbytes for img in target_intensity_images) / 1024**2:.1f} MB")

    try:
        config = IDTConfig()
        _print_idt_input_diagnostics(
            idt_params,
            target_intensity_images,
            ref_intensity_images,
            u_illumination_list,
            config,
        )
        n_re, _ = compute_idt(
            idt_params,
            target_intensity_images,
            ref_intensity_images,
            u_illumination_list,
            config=config,
        )
        print("IDT computation completed.")
    except Exception as e:
        print(f"IDT computation failed with error: {e}")
        print("Attempting to clear memory and continue with reduced parameters...")
        jax.clear_caches()  # type: ignore[no-untyped-call]
        gc.collect()
        raise

    # Convert to real refractive index
    n_reconstructed = n_re

    # Debug: Print transfer function status
    print("\nTransfer functions computed successfully for IDT reconstruction.")
    print("For detailed transfer function debugging, run debug_transfer_functions.py")

    # Visualization
    _visualize_results(n_reconstructed, target_intensity_images, delta_n)

    # Clear JAX compilation cache at the end to prevent memory accumulation
    jax.clear_caches()  # type: ignore[no-untyped-call]

    # Force garbage collection to clean up any remaining large arrays
    gc.collect()

    # visualize_synthetic_spectra_profiles()


# %%
# Execute the main function
if __name__ == "__main__":
    main()
