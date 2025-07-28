"""IDT with MLB simulation example.

This example demonstrates how to use Intensity Diffraction Tomography (IDT)
with Multi-layer Born (MLB) simulation for microscopy analysis.
"""

# pyright: reportPossiblyUnboundVariable=false, reportInvalidTypeForm=false

import gc
import typing
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from muscopy_mlbsim.hologram_generator import HologramGenerator
from muscopy_mlbsim.mlb import (
    MLBForward,
    MLBParameters,
    get_oblique_wave_fft,
    get_scatter_potential,
)
from tqdm import tqdm

from muscopy.cfg import ArrayPrecision
from muscopy.idt import IDTParameters, compute_idt, transfer_func_im, transfer_func_re

# Suppress JAX warnings about dtype conversion that can interfere with execution
warnings.filterwarnings("ignore", category=FutureWarning, message=".*scatter inputs have incompatible types.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")


class IntensityImageSetGenerator:
    """Generate intensity image sets using MLB simulation and muscopy_mlbsim.HologramGenerator."""

    def __init__(
        self,
        idt_params: IDTParameters,
        mlb_params: MLBParameters,
        precision: ArrayPrecision,
    ) -> None:
        self.idt_params = idt_params
        self.mlb_params = mlb_params
        self.precision = precision

        # Initialize MLB forward simulator and hologram generator
        self.mlb_forward = MLBForward(mlb_params)
        self.hologram_generator = HologramGenerator(mlb_params)

        # Configure off-axis position
        self.hologram_generator.set_offaxis_position(0, 0)

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

    def set_scattering_potential(self, potential: Array) -> None:
        """Set the 3D scattering potential for the sample.

        Parameters
        ----------
        potential : Array
            3D scattering potential array
        """
        self.mlb_forward.set_scattering_potential(potential)

    def generate_intensity_image_set(
        self,
    ) -> tuple[list[Array], list[Array]]:
        r"""Generate intensity image set with different illumination angles.

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
        intensity_image_shape = (2 * self.idt_params.aperturesize_px + 1, 2 * self.idt_params.aperturesize_px + 1)

        if self.u_illumination_list is None:
            self.u_illumination_list = []

        print("Generating intensity images...")
        for angle in tqdm(self.angles):
            # Calculate illumination wave vector components
            kx_ill = (
                self.idt_params.light_freq_px
                * self.idt_params.na_illumination
                / self.idt_params.n_sol
                * self.idt_params.k_per_px
                * np.cos(angle)
            )
            ky_ill = (
                self.idt_params.light_freq_px
                * self.idt_params.na_illumination
                / self.idt_params.n_sol
                * self.idt_params.k_per_px
                * np.sin(angle)
            )

            self.u_illumination_list.append((kx_ill / self.idt_params.k_per_px, ky_ill / self.idt_params.k_per_px))

            # Generate oblique illumination wave
            input_field_fft = get_oblique_wave_fft(
                self.mlb_params,
                float(kx_ill),
                float(ky_ill),
            )

            # Set input field and simulate forward scattering
            self.mlb_forward.set_input_field_fft(input_field_fft)
            output_field = self.mlb_forward.get_observation_field()

            # Generate hologram using muscopy_mlbsim.HologramGenerator
            self.hologram_generator.set_target_field(output_field)
            hologram = self.hologram_generator.generate_hologram(
                hologram_shape=intensity_image_shape,
                reference_amplitude=0,
                bit_depth=16,
                output_dtype="float32",  # Keep as float for processing
            )

            # Normalize hologram to [0, 1] range for better numerical stability
            hologram /= 65535.0

            target_intensity_images.append(hologram)

            # Generate reference hologram (no scattering)
            ref_field = jnp.fft.ifft2(jnp.fft.ifftshift(input_field_fft))
            self.hologram_generator.set_target_field(ref_field)
            ref_hologram = self.hologram_generator.generate_hologram(
                hologram_shape=intensity_image_shape,
                reference_amplitude=0,
                bit_depth=16,
                output_dtype="float32",
            )

            # Normalize reference hologram to [0, 1] range for better numerical stability
            ref_hologram /= 65535.0

            reference_intensity_images.append(ref_hologram)

        # Clean up temporary variables to free memory
        del hologram, ref_hologram
        gc.collect()

        return target_intensity_images, reference_intensity_images


def generate_sphere_potential(
    mlb_params: MLBParameters,
    radius_um: float,
    delta_n: float,
    center_offset: tuple[int, int, int] = (0, 0, 0),
    precision: ArrayPrecision | None = None,
) -> Array:
    """Generate spherical scattering potential.

    Returns
    -------
    Array
        3D scattering potential array
    """
    if precision is None:
        precision = ArrayPrecision()

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
        dtype=precision.float_precision(),
    )
    refractive_index = jnp.where(sphere_mask, n_sphere, refractive_index)

    # Convert to scattering potential
    return typing.cast("Array", get_scatter_potential(mlb_params, refractive_index))


def visualize_transfer_functions(
    idt_params: IDTParameters,
    u_illumination_list: list[tuple[float, float]],
    z_slice: float = 0.0,
    save_path: str | None = None,
) -> None:
    """Visualize transfer functions for debugging IDT implementation.

    Parameters
    ----------
    idt_params : IDTParameters
        IDT parameters for transfer function calculation
    u_illumination_list : list[tuple[float, float]]
        List of illumination angles in normalized frequency units
    z_slice : float, optional
        Z position for transfer function calculation, by default 0.0
    save_path : str | None, optional
        Path to save the visualization, by default None
    """
    print(f"Visualizing transfer functions at z={z_slice}...")

    # Select a subset of illumination angles for visualization
    n_angles_to_show = min(6, len(u_illumination_list))
    angles_to_show = u_illumination_list[:n_angles_to_show]

    # Create figure with subplots
    _, axes = plt.subplots(2, n_angles_to_show, figsize=(4 * n_angles_to_show, 8))
    if n_angles_to_show == 1:
        axes = axes.reshape(2, 1)

    for i, (u_x, u_y) in enumerate(angles_to_show):
        # Calculate transfer functions
        h_re = transfer_func_re(idt_params, (u_x, u_y), z_slice, 1.0)
        h_im = transfer_func_im(idt_params, (u_x, u_y), z_slice, 1.0)

        # Convert to numpy for plotting
        h_re_np = np.array(h_re) if hasattr(h_re, "__array__") else h_re
        h_im_np = np.array(h_im) if hasattr(h_im, "__array__") else h_im

        # Plot real part
        im_re = axes[0, i].imshow(np.abs(h_re_np), cmap="viridis", aspect="equal")
        axes[0, i].set_title(f"Real TF |H_re|\nAngle: ({u_x:.3f}, {u_y:.3f})")
        axes[0, i].set_xlabel("kx [px]")
        axes[0, i].set_ylabel("ky [px]")
        plt.colorbar(im_re, ax=axes[0, i])

        # Plot imaginary part
        im_im = axes[1, i].imshow(np.abs(h_im_np), cmap="plasma", aspect="equal")
        axes[1, i].set_title(f"Imag TF |H_im|\nAngle: ({u_x:.3f}, {u_y:.3f})")
        axes[1, i].set_xlabel("kx [px]")
        axes[1, i].set_ylabel("ky [px]")
        plt.colorbar(im_im, ax=axes[1, i])

        # Print statistics
        print(f"Angle ({u_x:.3f}, {u_y:.3f}):")
        print(f"  H_re: range=[{h_re_np.min():.2e}, {h_re_np.max():.2e}], mean={h_re_np.mean():.2e}")
        print(f"  H_im: range=[{h_im_np.min():.2e}, {h_im_np.max():.2e}], mean={h_im_np.mean():.2e}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Transfer function visualization saved to {save_path}")

    plt.show()
    plt.close()


def analyze_transfer_function_properties(
    idt_params: IDTParameters,
    u_illumination_list: list[tuple[float, float]],
    z_values: list[float] | None = None,
) -> None:
    """Analyze transfer function properties across different z positions and angles.

    Parameters
    ----------
    idt_params : IDTParameters
        IDT parameters
    u_illumination_list : list[tuple[float, float]]
        Illumination angles
    z_values : list[float] | None, optional
        Z positions to analyze, by default None (uses a default range)
    """
    if z_values is None:
        z_values = [0.0, idt_params.imgpx_axial_m_per_px * 5, idt_params.imgpx_axial_m_per_px * 10]

    print("Transfer Function Analysis:")
    print(f"IDT Parameters: NA={idt_params.na}, wavelength={idt_params.wavelength_m * 1e9:.0f}nm")
    print(f"Number of illumination angles: {len(u_illumination_list)}")

    # Analyze first few angles
    for i, (u_x, u_y) in enumerate(u_illumination_list[:3]):
        print(f"\nAngle {i + 1}: u=({u_x:.4f}, {u_y:.4f})")
        angle_mag = np.sqrt(u_x**2 + u_y**2)
        print(f"  Angle magnitude: {angle_mag:.4f} (max theoretical: {idt_params.na / idt_params.n_sol:.4f})")

        for z in z_values:
            h_re = transfer_func_re(idt_params, (u_x, u_y), z, 1.0)
            h_im = transfer_func_im(idt_params, (u_x, u_y), z, 1.0)

            h_re_np = np.array(h_re) if hasattr(h_re, "__array__") else h_re
            h_im_np = np.array(h_im) if hasattr(h_im, "__array__") else h_im

            re_max = np.abs(h_re_np).max()
            im_max = np.abs(h_im_np).max()
            print(f"  z={z * 1e6:.1f}μm: |H_re|_max={re_max:.2e}, |H_im|_max={im_max:.2e}")

            # Check for NaN or infinite values
            if np.any(np.isnan(h_re_np)) or np.any(np.isinf(h_re_np)):
                print("    WARNING: H_re contains NaN or Inf values!")
            if np.any(np.isnan(h_im_np)) or np.any(np.isinf(h_im_np)):
                print("    WARNING: H_im contains NaN or Inf values!")


def _setup_parameters() -> tuple[IDTParameters, MLBParameters, ArrayPrecision]:
    r"""Set up ODT and MLB simulation parameters.

    Returns
    -------
    tuple[IDTParameters, MLBParameters, ArrayPrecision]
        ODT parameters, MLB parameters, and array precision settings
    """
    # Use 32-bit precision to avoid JAX complex128 warnings (complex64 is sufficient)
    precision = ArrayPrecision(int_length=16, float_length=32)

    # IDT parameters - use more conservative values for stability and reduced memory usage
    print("Setting ODT parameters...")
    idt_params = IDTParameters(
        na=0.6,  # Reduced NA for stability
        wavelength_m=532e-9,  # 532 nm
        img_size_px=256,  # Reduced image size to decrease memory usage
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.3,  # Reduced illumination NA
        num_z_slices=128,  # Reduced z slices to decrease memory usage
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

    return idt_params, mlb_params, precision


def _generate_intensity_images(
    idt_params: IDTParameters,
    mlb_params: MLBParameters,
    scattering_potential: Array,
    precision: ArrayPrecision,
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
    # Create save directory if it doesn't exist
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_images_path = save_dir / "target_intensity_images.npy"
    ref_images_path = save_dir / "ref_intensity_images.npy"
    u_illumination_path = save_dir / "u_illumination_list.npy"

    # Check if saved images exist
    if target_images_path.exists() and ref_images_path.exists() and u_illumination_path.exists():
        print(f"Loading existing intensity images from {save_path}...")

        # Load saved images
        target_intensity_images_np = np.load(target_images_path)
        ref_intensity_images_np = np.load(ref_images_path)
        u_illumination_list_np = np.load(u_illumination_path)
        # Convert numpy arrays back to JAX arrays and lists
        target_intensity_images = [jnp.array(img) for img in target_intensity_images_np]
        ref_intensity_images = [jnp.array(img) for img in ref_intensity_images_np]
        u_illumination_list = [(float(u[0]), float(u[1])) for u in u_illumination_list_np]

        print(f"Loaded {len(target_intensity_images)} intensity images from disk.")
        return (target_intensity_images, ref_intensity_images), u_illumination_list

    # Generate new images if not found
    print("Setting up hologram generator...")
    intensity_image_gen = IntensityImageSetGenerator(idt_params, mlb_params, precision)
    intensity_image_gen.set_illumination_angles(num_angles)  # Fewer angles for faster computation
    intensity_image_gen.set_scattering_potential(scattering_potential)

    intensity_images = intensity_image_gen.generate_intensity_image_set()
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

    np.save(target_images_path, target_intensity_images_np)
    np.save(ref_images_path, ref_intensity_images_np)
    np.save(u_illumination_path, u_illumination_list_np)

    print(f"Saved {len(target_intensity_images)} intensity images to disk.")

    # Clean up the generator to free memory
    intensity_image_gen.cleanup()

    return (target_intensity_images, ref_intensity_images), u_illumination_list


def _visualize_results(
    n_reconstructed: Array,
    target_intensity_images: list[Array],
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
    hol_display = (
        np.array(target_intensity_images[0])
        if hasattr(target_intensity_images[0], "__array__")
        else target_intensity_images[0]
    )

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
    plt.close()  # Close the figure to free memory

    # Print summary
    print("\nReconstruction Summary:")
    print(f"Shape: {n_reconstructed_np.shape}")
    print(f"Δn range: [{n_reconstructed_np.min():.4f}, {n_reconstructed_np.max():.4f}]")
    print(f"Expected Δn: {delta_n:.4f}")
    print(f"Peak Δn: {n_reconstructed_np.max():.4f}")
    print(f"Recovery ratio: {n_reconstructed_np.max() / delta_n:.2f}")


def main() -> None:
    """Demonstrate IDT with MLB simulation."""
    # Clear JAX compilation cache at the start to prevent memory accumulation
    jax.clear_caches()  # type: ignore[no-untyped-call]

    # Setup parameters
    idt_params, mlb_params, precision = _setup_parameters()
    num_angles = 12  # Reduced number of angles to decrease computation time and memory usage

    # Generate sample (sphere) - increase scattering for better signal
    print("Generating spherical sample...")
    radius_um = 3.0  # Larger sphere
    delta_n = 0.1  # Much stronger scattering
    scattering_potential = generate_sphere_potential(mlb_params, radius_um, delta_n, precision=precision)

    print(f"Sample size: {scattering_potential.shape}")
    print(f"Memory usage: {scattering_potential.nbytes / 1024**2:.1f} MB")

    # Generate holograms (or load from disk if available)
    print("Starting hologram generation...")
    (target_intensity_images, ref_intensity_images), u_illumination_list = _generate_intensity_images(
        idt_params, mlb_params, scattering_potential, precision, num_angles, save_path="intensity_images"
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
        n_re, _ = compute_idt(idt_params, target_intensity_images, ref_intensity_images, u_illumination_list)
        print("IDT computation completed.")
    except Exception as e:
        print(f"IDT computation failed with error: {e}")
        print("Attempting to clear memory and continue with reduced parameters...")
        jax.clear_caches()  # type: ignore[no-untyped-call]
        gc.collect()
        raise

    # Convert to real refractive index
    n_reconstructed = n_re - idt_params.n_sol

    # Debug: Analyze transfer functions before visualization
    print("\n" + "=" * 60)
    print("TRANSFER FUNCTION DEBUGGING")
    print("=" * 60)

    analyze_transfer_function_properties(idt_params, u_illumination_list)

    # Visualize transfer functions for the first few z slices
    z_debug_positions = [0.0, idt_params.imgpx_axial_m_per_px * 5]
    for z_pos in z_debug_positions:
        visualize_transfer_functions(
            idt_params, u_illumination_list, z_slice=z_pos, save_path=f"transfer_functions_z{z_pos * 1e6:.1f}um.png"
        )

    print("=" * 60)
    print("END TRANSFER FUNCTION DEBUGGING")
    print("=" * 60 + "\n")

    # Visualization
    _visualize_results(n_reconstructed, target_intensity_images, delta_n)

    # Clear JAX compilation cache at the end to prevent memory accumulation
    jax.clear_caches()  # type: ignore[no-untyped-call]

    # Force garbage collection to clean up any remaining large arrays
    gc.collect()

    # visualize_synthetic_spectra_profiles()


if __name__ == "__main__":
    main()
