"""
Demonstration of ODT reconstruction using MLB simulation
========================================================

This example demonstrates how to use the `muscopy.odt` function combined with
muscopy_mlbsim to perform Optical Diffraction Tomography (ODT) reconstruction
from synthetic holograms generated using Multi-layer Born (MLB) forward simulation.

The workflow includes:
1. Setting up simulation parameters for Multi-layer Born forward model
2. Generating synthetic holograms using muscopy_mlbsim HologramGenerator
3. Performing ODT reconstruction using muscopy
4. Visualizing and analyzing the reconstruction results

Requirements:
- The private muscopy_mlbsim package must be installed separately
- GPU support (JAX) is recommended for faster computation

See :ref:`mlb-installation` for repository access and installation instructions.
Without the simulator, this example skips execution.
"""

from importlib.util import find_spec

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from muscopy.cfg import ArrayPrecision
from muscopy.dh import get_spectrum
from muscopy.example_mlb import simulate_mlb_measurements
from muscopy.odt import ODTConfig, ODTParameters, odt


def _extract_spectra(
    target_holograms: list[Array],
    ref_holograms: list[Array],
    odt_params: ODTParameters,
    offaxis_center: tuple[int, int],
) -> tuple[list[Array], list[Array]]:
    r"""Extract complex field spectra from holograms.

    Returns
    -------
    tuple[list[Array], list[Array]]
        Complex field spectra and reference spectra
    """
    print("Extracting complex field spectra...")
    cp_spectrums = []
    ref_cp_spectrums = []

    for target_hol, ref_hol in zip(target_holograms, ref_holograms, strict=True):
        # Convert to complex field spectrum
        ft_target = jnp.fft.fftshift(jnp.fft.fft2(target_hol))
        ft_ref = jnp.fft.fftshift(jnp.fft.fft2(ref_hol))

        # Extract complex field using get_spectrum
        cp_spectrum = get_spectrum(ft_target, odt_params, offaxis_center)
        ref_cp_spectrum = get_spectrum(ft_ref, odt_params, offaxis_center)

        cp_spectrums.append(cp_spectrum)
        ref_cp_spectrums.append(ref_cp_spectrum)

    return cp_spectrums, ref_cp_spectrums


def _visualize_results(
    n_reconstructed: Array,
    target_holograms: list[Array],
    delta_n: float,
) -> None:
    """Create visualizations of the reconstruction results."""
    print("Creating visualizations...")

    # Convert to numpy for matplotlib
    n_reconstructed_np = np.array(n_reconstructed) if hasattr(n_reconstructed, "__array__") else n_reconstructed

    # Cross-sections through the center
    center_x = n_reconstructed_np.shape[0] // 2
    center_y = n_reconstructed_np.shape[1] // 2
    center_z = n_reconstructed_np.shape[2] // 2

    # Create figure with subplots
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original hologram
    hol_display = np.array(target_holograms[0]) if hasattr(target_holograms[0], "__array__") else target_holograms[0]

    im1 = axes[0, 0].imshow(hol_display, cmap="gray")
    axes[0, 0].set_title("Sample Hologram")
    axes[0, 0].set_xlabel("x [px]")
    axes[0, 0].set_ylabel("y [px]")
    plt.colorbar(im1, ax=axes[0, 0])

    # Cross-sections of reconstruction
    im2 = axes[0, 1].imshow(n_reconstructed_np[:, :, center_z], cmap="viridis")
    axes[0, 1].set_title("XY Cross-section (Center Z)")
    axes[0, 1].set_xlabel("x [px]")
    axes[0, 1].set_ylabel("y [px]")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(n_reconstructed_np[:, center_y, :], cmap="viridis")
    axes[0, 2].set_title("XZ Cross-section (Center Y)")
    axes[0, 2].set_xlabel("z [px]")
    axes[0, 2].set_ylabel("x [px]")
    plt.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(n_reconstructed_np[center_x, :, :], cmap="viridis")
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
    axes[1, 1].grid(True)

    # Show max projection
    max_proj = np.max(n_reconstructed_np, axis=2)
    im6 = axes[1, 2].imshow(max_proj, cmap="viridis")
    axes[1, 2].set_title("Maximum Projection (Z axis)")
    axes[1, 2].set_xlabel("x [px]")
    axes[1, 2].set_ylabel("y [px]")
    plt.colorbar(im6, ax=axes[1, 2])

    plt.tight_layout()
    plt.savefig("odt_mlb_simulation_results.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\nReconstruction Summary:")
    print(f"Shape: {n_reconstructed_np.shape}")
    print(f"Δn range: [{n_reconstructed_np.min():.4f}, {n_reconstructed_np.max():.4f}]")
    print(f"Expected Δn: {delta_n:.4f}")
    print(f"Peak Δn: {n_reconstructed_np.max():.4f}")
    print(f"Recovery ratio: {n_reconstructed_np.max() / delta_n:.2f}")


def visualize_synthetic_spectra_profiles(synthetic_spectra: Array) -> None:  # noqa: PLR0914, PLR0915
    """Visualize XY and XZ profiles of synthetic_spectra with log scale.

    Parameters
    ----------
    synthetic_spectra : Array
        3D complex array with shape (x, y, z)
    """
    print("Visualizing synthetic spectra profiles...")

    # Convert to numpy for matplotlib and take log of absolute values
    spectra_np = np.array(synthetic_spectra) if hasattr(synthetic_spectra, "__array__") else synthetic_spectra
    log_abs_spectra = np.log(np.abs(spectra_np) + 1e-12)  # Add small value to avoid log(0)

    # Get array dimensions and print for debugging
    print(f"Synthetic spectra shape: {log_abs_spectra.shape}")

    nx, ny, nz = log_abs_spectra.shape
    angle_data = log_abs_spectra
    center_x = nx // 2
    center_y = ny // 2
    center_z = nz // 2

    # Create figure with subplots
    _, axes = plt.subplots(2, 3, figsize=(15, 10))

    # XY profile (center Z)
    xy_profile = angle_data[:, :, center_z]
    im1 = axes[0, 0].imshow(xy_profile, cmap="viridis", aspect="equal")
    axes[0, 0].set_title(f"XY Profile (Z={center_z})")
    axes[0, 0].set_xlabel("Y [px]")
    axes[0, 0].set_ylabel("X [px]")
    plt.colorbar(im1, ax=axes[0, 0], label="log|amplitude|")

    # XZ profile (center Y)
    xz_profile = angle_data[:, center_y, :]
    im2 = axes[0, 1].imshow(xz_profile, cmap="viridis", aspect="equal")
    axes[0, 1].set_title(f"XZ Profile (Y={center_y})")
    axes[0, 1].set_xlabel("Z [px]")
    axes[0, 1].set_ylabel("X [px]")
    plt.colorbar(im2, ax=axes[0, 1], label="log|amplitude|")

    # YZ profile (center X)
    yz_profile = angle_data[center_x, :, :]
    im3 = axes[0, 2].imshow(yz_profile, cmap="viridis", aspect="equal")
    axes[0, 2].set_title(f"YZ Profile (X={center_x})")
    axes[0, 2].set_xlabel("Z [px]")
    axes[0, 2].set_ylabel("Y [px]")
    plt.colorbar(im3, ax=axes[0, 2], label="log|amplitude|")

    # Line profiles
    # X direction through center
    x_line = angle_data[:, center_y, center_z]
    axes[1, 0].plot(x_line)
    axes[1, 0].set_title("X-direction Profile")
    axes[1, 0].set_xlabel("X [px]")
    axes[1, 0].set_ylabel("log|amplitude|")
    axes[1, 0].grid(True)

    # Y direction through center
    y_line = angle_data[center_x, :, center_z]
    axes[1, 1].plot(y_line)
    axes[1, 1].set_title("Y-direction Profile")
    axes[1, 1].set_xlabel("Y [px]")
    axes[1, 1].set_ylabel("log|amplitude|")
    axes[1, 1].grid(True)

    # Z direction through center
    z_line = angle_data[center_x, center_y, :]
    axes[1, 2].plot(z_line)
    axes[1, 2].set_title("Z-direction Profile")
    axes[1, 2].set_xlabel("Z [px]")
    axes[1, 2].set_ylabel("log|amplitude|")
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig("synthetic_spectra_profiles.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print("\nSynthetic Spectra Summary:")
    print(f"Shape: {spectra_np.shape}")
    print(f"Data type: {spectra_np.dtype}")
    print(f"Log|amplitude| range: [{log_abs_spectra.min():.4f}, {log_abs_spectra.max():.4f}]")
    print(f"Original |amplitude| range: [{np.abs(spectra_np).min():.2e}, {np.abs(spectra_np).max():.2e}]")
    print(f"3D volume shape: {log_abs_spectra.shape}")


# %%
# Main execution
def main() -> None:
    """Demonstrate ODT with MLB simulation."""
    if find_spec("muscopy_mlbsim") is None:
        print(
            "Skipping ODT demo because muscopy-mlbsim is unavailable. See "
            "https://muscopy.readthedocs.io/en/latest/mlb_installation.html for installation instructions."
        )
        return

    odt_params = ODTParameters(
        na=1.1,
        wavelength_m=532e-9,
        img_size_px=512,
        px_size_m=3.45e-6 * 3 / 180 / 2,
        n_sol=1.33,
        na_illumination=1.0,
    )
    offaxis_center = (200, 200)
    opposite_offaxis_center = (odt_params.img_size_px - offaxis_center[0], odt_params.img_size_px - offaxis_center[1])
    delta_n = 0.02
    target_holograms, ref_holograms, _ = simulate_mlb_measurements(
        odt_params, radius_um=2.0, delta_n=delta_n, num_angles=10, off_axis_center=offaxis_center
    )

    # Extract complex field spectra
    cp_spectrums, ref_cp_spectrums = _extract_spectra(
        target_holograms, ref_holograms, odt_params, opposite_offaxis_center
    )

    # ODT reconstruction
    print("Performing ODT reconstruction...")
    odt_config = ODTConfig(
        approx_type="Rytov",
        hermite_symmetry=True,
        precision=ArrayPrecision(),
        edge_size=8,
    )

    refractive_index, synthetic_spectra = odt(cp_spectrums, ref_cp_spectrums, odt_params, odt_config)

    # Convert to real refractive index
    n_reconstructed = jnp.real(refractive_index) - odt_params.n_sol

    # Visualization
    _visualize_results(n_reconstructed, target_holograms, delta_n)

    # Visualize synthetic spectra profiles
    visualize_synthetic_spectra_profiles(synthetic_spectra)


# %%
# Execute the main function
if __name__ == "__main__":
    main()
