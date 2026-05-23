"""
IDT with MLB Simulation Example
===============================

This example demonstrates how to use Intensity Diffraction Tomography (IDT)
with Multi-layer Born (MLB) simulation for microscopy analysis.
"""

# pyright: reportPossiblyUnboundVariable=false, reportInvalidTypeForm=false

import gc
import shutil
import typing
import warnings
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from tqdm import tqdm

from muscopy.idt import IDTConfig, IDTParameters, compute_g_list, compute_idt, transfer_func_im, transfer_func_re

try:
    from muscopy_mlbsim import (  # pyright: ignore[reportMissingImports]
        MLBForward,
        MLBParameters,
        get_oblique_wave_fft,
        get_scatter_potential,
    )

    MLB_AVAILABLE = True
except ImportError:
    MLB_AVAILABLE = False
    print("Warning: muscopy_mlbsim is not installed. This example requires muscopy_mlbsim.")
    print("Skipping example execution.")
    MLBForward: typing.Any = None  # type: ignore[no-redef]
    MLBParameters: typing.Any = None  # type: ignore[no-redef]
    get_oblique_wave_fft: typing.Any = None  # type: ignore[no-redef]
    get_scatter_potential: typing.Any = None  # type: ignore[no-redef]

# Suppress JAX warnings about dtype conversion that can interfere with execution
warnings.filterwarnings("ignore", category=FutureWarning, message=".*scatter inputs have incompatible types.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")

INTENSITY_IMAGE_SIZE = 1023  # Increased for better quality
_DIAGNOSTIC_EPSILON = 1e-6


def _field_to_intensity_image(field: Array, mlb_params: MLBParameters, image_shape: tuple[int, int]) -> Array:
    """Convert a detector field to a padded intensity image without per-frame normalization.

    Returns
    -------
    Array
        Padded detector intensity image.

    Raises
    ------
    ValueError
        If the requested image shape is smaller than the MLB field shape.
    """
    if image_shape[0] < mlb_params.xy_shape[0] or image_shape[1] < mlb_params.xy_shape[1]:
        msg = "image_shape must be at least as large as the MLB lateral field shape."
        raise ValueError(msg)

    field_fft = jnp.fft.fftshift(jnp.fft.fft2(field))
    image_fft = jnp.zeros(image_shape, dtype=field_fft.dtype)
    x_start = image_shape[0] // 2 - mlb_params.xy_shape[0] // 2
    y_start = image_shape[1] // 2 - mlb_params.xy_shape[1] // 2
    image_fft = image_fft.at[
        x_start : x_start + mlb_params.xy_shape[0],
        y_start : y_start + mlb_params.xy_shape[1],
    ].set(field_fft)

    image_field = jnp.fft.ifft2(jnp.fft.ifftshift(image_fft))
    return jnp.asarray(jnp.abs(image_field) ** 2, dtype=jnp.float32)


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


class IntensityImageSetGenerator:
    """Generate intensity image sets using MLB simulation."""

    def __init__(
        self,
        idt_params: IDTParameters,
        mlb_params: MLBParameters,
    ) -> None:
        if not MLB_AVAILABLE:
            msg = "muscopy_mlbsim is required but not available"
            raise ImportError(msg)

        self.idt_params = idt_params
        self.mlb_params = mlb_params

        # Initialize MLB forward simulator
        self.mlb_forward = MLBForward(mlb_params)

        # Store illumination angles
        self.angles: np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]] | None = None
        self.u_illumination_list: list[tuple[float, float]] | None = None

    def cleanup(self) -> None:
        """Clean up resources and clear cached data."""
        # Clear any cached data in the MLB forward simulator and hologram generator
        self.angles = None
        self.u_illumination_list = None

        # Force garbage collection to clean up large arrays
        gc.collect()

    def set_illumination_angles(self, num_angles: int, angle_offset: float = 0) -> None:
        """Set illumination angles for tomographic acquisition."""
        self.num_angles = num_angles
        self.angle_offset = angle_offset
        self.angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False) + angle_offset

    def generate_intensity_image_set(
        self,
        scattering_potential: Array,
    ) -> tuple[list[Array], list[Array]]:
        r"""Generate intensity image set with different illumination angles.

        Parameters
        ----------
        scattering_potential : Array
            3D scattering potential array

        Returns
        -------
        tuple[list[Array], list[Array]]
            Target intensity images and reference intensity images

        Raises
        ------
        ValueError
            If illumination angles are not set
        """
        if self.angles is None:
            msg = "Illumination angles not set. Call set_illumination_angles() first."
            raise ValueError(msg)

        target_intensity_images = []
        reference_intensity_images = []
        # Use aperture size for consistency with IDT calculations
        intensity_image_shape = (INTENSITY_IMAGE_SIZE, INTENSITY_IMAGE_SIZE)

        if self.u_illumination_list is None:
            self.u_illumination_list = []

        print("Generating intensity images...")
        for angle in tqdm(self.angles):
            # Calculate illumination wave vector components
            kx_ill = (
                self.idt_params.light_freq_px * self.idt_params.na_illumination / self.idt_params.n_sol * np.cos(angle)
            )
            ky_ill = (
                self.idt_params.light_freq_px * self.idt_params.na_illumination / self.idt_params.n_sol * np.sin(angle)
            )

            self.u_illumination_list.append((kx_ill, ky_ill))

            # Generate oblique illumination wave
            input_field_fft = get_oblique_wave_fft(
                self.mlb_params,
                float(kx_ill * self.idt_params.k_per_px),
                float(ky_ill * self.idt_params.k_per_px),
            )

            # Simulate the forward-scattered field at the detector plane
            output_field = self.mlb_forward.simulate_forward_detector_field(input_field_fft, scattering_potential)

            target_intensity_images.append(
                _field_to_intensity_image(output_field, self.mlb_params, intensity_image_shape)
            )

            # Generate reference hologram (no scattering)
            ref_field = self.mlb_forward.propagate_forward_to_detector(input_field_fft)
            reference_intensity_images.append(
                _field_to_intensity_image(ref_field, self.mlb_params, intensity_image_shape)
            )

        gc.collect()

        return target_intensity_images, reference_intensity_images


def generate_sphere_potential(
    mlb_params: MLBParameters,
    radius_um: float,
    delta_n: float,
    center_offset: tuple[int, int, int] = (0, 0, 0),
) -> Array:
    """Generate spherical scattering potential.

    Returns
    -------
    Array
        3D scattering potential array
    """
    # Convert radius from micrometers to pixels
    radius_px = int(radius_um * 1e-6 / mlb_params.dxy_m)

    # Create coordinate grids
    z, y, x = jnp.meshgrid(
        jnp.arange(mlb_params.num_layers),
        jnp.arange(mlb_params.xy_shape[0]),
        jnp.arange(mlb_params.xy_shape[1]),
        indexing="ij",
    )

    # Sphere center
    center_z = mlb_params.num_layers // 2 + center_offset[0]
    center_y = mlb_params.xy_shape[0] // 2 + center_offset[1]
    center_x = mlb_params.xy_shape[1] // 2 + center_offset[2]

    # Create sphere mask
    sphere_mask = ((z - center_z) ** 2 + (y - center_y) ** 2 + (x - center_x) ** 2) < radius_px**2

    # Create refractive index distribution
    n_background = mlb_params.n_medium
    n_sphere = n_background + delta_n

    refractive_index = jnp.full(
        (mlb_params.num_layers, mlb_params.xy_shape[0], mlb_params.xy_shape[1]),
        n_background,
        dtype=jnp.float32,
    )
    refractive_index = jnp.where(sphere_mask, n_sphere, refractive_index)

    # Convert to scattering potential
    return jnp.asarray(get_scatter_potential(mlb_params, refractive_index))


def _setup_parameters() -> tuple[IDTParameters, MLBParameters]:
    r"""Set up IDT and MLB simulation parameters.

    Returns
    -------
    tuple[IDTParameters, MLBParameters]
        IDT parameters and MLB parameters
    """
    # IDT parameters - use more conservative values for stability and reduced memory usage
    print("Setting IDT parameters...")
    idt_params = IDTParameters(
        na=0.8,
        wavelength_m=532e-9,  # 532 nm
        img_size_px=INTENSITY_IMAGE_SIZE,
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.6,
        num_z_slices=64,  # Increased for better z-resolution
    )

    # MLB simulation parameters
    print("Setting MLB simulation parameters...")
    mlb_params = MLBParameters(
        wavelength_m=idt_params.wavelength_m,
        numerical_aperture=idt_params.na,
        n_medium=idt_params.n_sol,
        xy_shape=(2 * idt_params.aperturesize_px, 2 * idt_params.aperturesize_px),
        num_layers=idt_params.freq_axial_extent_px,
        dxy_m=idt_params.imgpx_lateral_m_per_px,
        dz_m=idt_params.imgpx_axial_m_per_px,
    )

    return idt_params, mlb_params


def _generate_intensity_images(
    idt_params: IDTParameters,
    mlb_params: MLBParameters,
    scattering_potential: Array,
    num_angles: int = 60,
    save_path: str = "intensity_images",
) -> tuple[tuple[list[Array], list[Array]], list[tuple[float, float]]]:
    r"""Generate hologram sets.

    Parameters
    ----------
    save_path : str, optional
        Directory to save/load intensity images, by default "intensity_images"

    Returns
    -------
    tuple[tuple[list[Array], list[Array]], list[tuple[float, float]]]
        Target holograms, reference holograms, and illumination angles
    """
    # Delete existing saved images if they exist
    save_dir = Path(save_path)
    if save_dir.exists():
        print(f"Removing existing intensity images at {save_path}...")
        shutil.rmtree(save_dir)

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Always generate new images
    print("Setting up hologram generator...")
    intensity_image_gen = IntensityImageSetGenerator(idt_params, mlb_params)
    intensity_image_gen.set_illumination_angles(num_angles)  # Fewer angles for faster computation

    intensity_images = intensity_image_gen.generate_intensity_image_set(scattering_potential)
    target_intensity_images, ref_intensity_images = intensity_images

    u_illumination_list = intensity_image_gen.u_illumination_list
    if u_illumination_list is None:
        u_illumination_list = []

    # Save generated images to disk
    print(f"Saving intensity images to {save_path}...")

    # Convert JAX arrays to numpy arrays for saving
    target_intensity_images_np = np.array([np.array(img) for img in target_intensity_images])
    ref_intensity_images_np = np.array([np.array(img) for img in ref_intensity_images])
    u_illumination_list_np = np.array(u_illumination_list)

    target_images_path = save_dir / "target_intensity_images.npy"
    ref_images_path = save_dir / "ref_intensity_images.npy"
    u_illumination_path = save_dir / "u_illumination_list.npy"

    np.save(target_images_path, target_intensity_images_np)
    np.save(ref_images_path, ref_intensity_images_np)
    np.save(u_illumination_path, u_illumination_list_np)

    print(f"Generated and saved {len(target_intensity_images)} intensity images.")

    # Clean up the generator to free memory
    intensity_image_gen.cleanup()

    return (target_intensity_images, ref_intensity_images), u_illumination_list


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
    if not MLB_AVAILABLE:
        print("Skipping IDT with MLB simulation demo - muscopy_mlbsim not available.")
        # Create a simple placeholder plot for documentation
        _, ax = plt.subplots(figsize=(8, 6))
        message = (
            "muscopy_mlbsim Required\\n\\n"
            "This example requires the muscopy_mlbsim package.\\n"
            "Please install or upgrade muscopy_mlbsim before running this example."
        )
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "lightgray"},
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        plt.title("IDT with MLB Simulation Example")
        plt.tight_layout()
        plt.savefig("idt_mlb_simulation_placeholder.png", dpi=150, bbox_inches="tight")
        print("Placeholder visualization saved to idt_mlb_simulation_placeholder.png")
        plt.close()
        return

    # Clear JAX compilation cache at the start to prevent memory accumulation
    jax.clear_caches()  # type: ignore[no-untyped-call]

    # Setup parameters
    idt_params, mlb_params = _setup_parameters()
    num_angles = 10  # Reduced number of angles to decrease computation time and memory usage

    # Generate sample (sphere) - increase scattering for better signal
    print("Generating spherical sample...")
    radius_um = 3.0  # Larger sphere
    delta_n = 0.01  # Much stronger scattering
    scattering_potential = generate_sphere_potential(mlb_params, radius_um, delta_n)

    # SlicingVisualizer(np.asarray(scattering_potential)).run()

    print(f"Sample size: {scattering_potential.shape}")
    print(f"Memory usage: {scattering_potential.nbytes / 1024**2:.1f} MB")

    # Generate holograms (or load from disk if available)
    print("Starting hologram generation...")
    (target_intensity_images, ref_intensity_images), u_illumination_list = _generate_intensity_images(
        idt_params, mlb_params, scattering_potential, num_angles, save_path="intensity_images"
    )
    print(f"Hologram generation completed. Using {len(target_intensity_images)} holograms.")

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
