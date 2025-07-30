"""Test intensity image generation for MLB simulation.

This script tests the intensity image generation process in isolation
to debug issues with hologram generation.
"""

import gc
import typing
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from ilabvis.slice_visualizer import SlicingVisualizer  # pyright: ignore[reportMissingImports]
from jax import Array
from muscopy_mlbsim.hologram_generator import HologramGenerator  # pyright: ignore[reportMissingImports]
from muscopy_mlbsim.mlb import (  # pyright: ignore[reportMissingImports]
    MLBForward,
    MLBParameters,
    get_oblique_wave_fft,
    get_scatter_potential,
)

from muscopy.cfg import ArrayPrecision
from muscopy.idt import IDTParameters

# Suppress JAX warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*scatter inputs have incompatible types.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Casting complex values to real.*")


def setup_test_parameters() -> tuple[IDTParameters, MLBParameters, ArrayPrecision]:
    """Set up simplified parameters for testing.

    Returns
    -------
    tuple[IDTParameters, MLBParameters, ArrayPrecision]
        Test parameters for IDT, MLB, and array precision
    """
    precision = ArrayPrecision(int_length=16, float_length=32)

    # Simplified IDT parameters
    idt_params = IDTParameters(
        na=0.6,
        wavelength_m=532e-9,
        img_size_px=1440,  # Smaller for testing
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.4,
        num_z_slices=256,  # Smaller for testing
    )

    # MLB simulation parameters
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


def create_simple_sphere_potential(
    mlb_params: MLBParameters,
    radius_um: float = 2.0,
    delta_n: float = 0.05,
) -> Array:
    """Create a simple spherical scattering potential for testing.

    Returns
    -------
    Array
        3D scattering potential array
    """
    print(f"Creating sphere potential: radius={radius_um}μm, delta_n={delta_n}")

    # Convert radius from micrometers to pixels
    radius_px = int(radius_um * 1e-6 / mlb_params.dxy_m)
    print(f"Radius in pixels: {radius_px}")

    # Create coordinate grids
    z, y, x = jnp.meshgrid(
        jnp.arange(mlb_params.num_layers),
        jnp.arange(mlb_params.xy_shape[0]),
        jnp.arange(mlb_params.xy_shape[1]),
        indexing="ij",
    )

    # Sphere center
    center_z = mlb_params.num_layers // 2
    center_y = mlb_params.xy_shape[0] // 2
    center_x = mlb_params.xy_shape[1] // 2

    print(f"Sphere center: ({center_z}, {center_y}, {center_x})")
    print(f"Volume shape: {mlb_params.num_layers} x {mlb_params.xy_shape}")

    # Create sphere mask
    sphere_mask = ((z - center_z) ** 2 + (y - center_y) ** 2 + (x - center_x) ** 2) < radius_px**2

    # Count voxels in sphere
    voxels_in_sphere = jnp.sum(sphere_mask)
    print(f"Voxels in sphere: {voxels_in_sphere}")

    # Create refractive index distribution
    n_background = mlb_params.n_medium
    n_sphere = n_background + delta_n

    refractive_index = jnp.full(
        (mlb_params.num_layers, mlb_params.xy_shape[0], mlb_params.xy_shape[1]),
        n_background,
        dtype=jnp.float32,
    )
    refractive_index = jnp.where(sphere_mask, n_sphere, refractive_index)

    refractive_index_arr = refractive_index
    print(f"Refractive index range: [{refractive_index_arr.min():.4f}, {refractive_index_arr.max():.4f}]")

    # Convert to scattering potential
    potential = get_scatter_potential(mlb_params, refractive_index)
    potential_arr = typing.cast("Array", potential)
    print(f"Scattering potential range: [{potential_arr.min():.2e}, {potential_arr.max():.2e}]")

    return typing.cast("Array", potential)


def test_single_intensity_image(  # noqa: PLR0915, PLR0914
    idt_params: IDTParameters,
    mlb_params: MLBParameters,
    scattering_potential: Array,
    angle: float,
    save_debug: bool,
) -> tuple[Array, Array]:
    """Test generation of a single intensity image.

    Returns
    -------
    tuple[Array, Array]
        Target and reference hologram arrays
    """
    print(f"\nTesting intensity image generation for angle: {angle:.2f} rad")

    # Initialize MLB forward simulator and hologram generator
    mlb_forward = MLBForward(mlb_params)
    hologram_generator = HologramGenerator(mlb_params)
    hologram_generator.set_offaxis_position(0, 0)

    # Set scattering potential
    mlb_forward.set_scattering_potential(scattering_potential)

    # Calculate illumination wave vector components
    kx_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.cos(angle)
    ky_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.sin(angle)

    print(f"Illumination k-vector: ({kx_ill:.4f}, {ky_ill:.4f})")

    # Generate oblique illumination wave
    input_field_fft = get_oblique_wave_fft(
        mlb_params,
        float(kx_ill * idt_params.k_per_px),
        float(ky_ill * idt_params.k_per_px),
    )

    print(f"Input field FFT shape: {input_field_fft.shape}")
    print(f"Input field FFT range: [{jnp.abs(input_field_fft).min():.2e}, {jnp.abs(input_field_fft).max():.2e}]")

    # Set input field and simulate forward scattering
    mlb_forward.set_input_field_fft(input_field_fft)
    output_field = mlb_forward.get_observation_field()

    print(f"Output field shape: {output_field.shape}")
    print(f"Output field range: [{jnp.abs(output_field).min():.2e}, {jnp.abs(output_field).max():.2e}]")

    # Use aperture size for consistency with IDT calculations
    intensity_image_shape = (1440, 1440)
    print(f"Target intensity image shape: {intensity_image_shape}")

    # Generate hologram using muscopy_mlbsim.HologramGenerator
    hologram_generator.set_target_field(output_field)
    hologram = hologram_generator.generate_hologram(
        hologram_shape=intensity_image_shape,
        reference_amplitude=0,
        bit_depth=16,
        output_dtype="float32",
    )

    print(f"Generated hologram shape: {hologram.shape}")
    print(f"Generated hologram range: [{hologram.min():.2e}, {hologram.max():.2e}]")

    # Normalize hologram to [0, 1] range
    hologram_normalized = hologram / 65535.0

    # Generate reference hologram (no scattering)
    ref_field = jnp.fft.ifft2(jnp.fft.ifftshift(input_field_fft))
    hologram_generator.set_target_field(ref_field)
    ref_hologram = hologram_generator.generate_hologram(
        hologram_shape=intensity_image_shape,
        reference_amplitude=0,
        bit_depth=16,
        output_dtype="float32",
    )

    ref_hologram_normalized = ref_hologram / 65535.0

    print(f"Reference hologram range: [{ref_hologram.min():.2e}, {ref_hologram.max():.2e}]")

    if save_debug:
        # Save debug information
        debug_dir = Path("intensity_debug")
        debug_dir.mkdir(exist_ok=True)

        # Save intermediate results
        np.save(debug_dir / f"input_field_fft_angle_{angle:.2f}.npy", np.array(input_field_fft))
        np.save(debug_dir / f"output_field_angle_{angle:.2f}.npy", np.array(output_field))
        np.save(debug_dir / f"hologram_angle_{angle:.2f}.npy", np.array(hologram))
        np.save(debug_dir / f"ref_hologram_angle_{angle:.2f}.npy", np.array(ref_hologram))

        # Create visualization
        _, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Input field magnitude
        input_field = jnp.fft.ifft2(jnp.fft.ifftshift(input_field_fft))
        im1 = axes[0, 0].imshow(np.abs(np.array(input_field)), cmap="viridis")
        axes[0, 0].set_title(f"Input Field |E| (angle={angle:.2f})")
        plt.colorbar(im1, ax=axes[0, 0])

        # Output field magnitude
        im2 = axes[0, 1].imshow(np.abs(np.array(output_field)), cmap="viridis")
        axes[0, 1].set_title("Output Field |E|")
        plt.colorbar(im2, ax=axes[0, 1])

        # Generated hologram
        im3 = axes[0, 2].imshow(np.array(hologram), cmap="gray")
        axes[0, 2].set_title("Generated Hologram")
        plt.colorbar(im3, ax=axes[0, 2])

        # Reference hologram
        im4 = axes[1, 0].imshow(np.array(ref_hologram), cmap="gray")
        axes[1, 0].set_title("Reference Hologram")
        plt.colorbar(im4, ax=axes[1, 0])

        # Difference
        diff = np.array(hologram) - np.array(ref_hologram)
        im5 = axes[1, 1].imshow(diff, cmap="RdBu_r")
        axes[1, 1].set_title("Hologram - Reference")
        plt.colorbar(im5, ax=axes[1, 1])

        # Central line profiles
        center = hologram.shape[0] // 2
        axes[1, 2].plot(np.array(hologram)[center, :], label="Hologram", alpha=0.7)
        axes[1, 2].plot(np.array(ref_hologram)[center, :], label="Reference", alpha=0.7)
        axes[1, 2].set_title("Central Line Profiles")
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plt.savefig(debug_dir / f"intensity_debug_angle_{angle:.2f}.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Debug information saved to {debug_dir}/")

    return hologram_normalized, ref_hologram_normalized


def test_multiple_angles(  # noqa: PLR0914
    idt_params: IDTParameters,
    mlb_params: MLBParameters,
    scattering_potential: Array,
    num_angles: int,
    save_npy: bool,
) -> None:
    """Test generation of multiple intensity images."""
    print(f"\nTesting {num_angles} intensity images...")

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    target_images = []
    ref_images = []
    u_illumination_list = []

    for i, angle in enumerate(angles):
        print(f"\nProcessing angle {i + 1}/{num_angles}: {angle:.2f} rad")

        # Generate intensity images
        hologram, ref_hologram = test_single_intensity_image(
            idt_params, mlb_params, scattering_potential, float(angle), (i == 0)
        )

        target_images.append(hologram)
        ref_images.append(ref_hologram)

        # Calculate u_illumination for IDT
        kx_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.cos(angle)
        ky_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.sin(angle)
        u_illumination_list.append((kx_ill, ky_ill))

    if save_npy:
        # Save in the same format as the main script
        output_dir = Path("intensity_images_test")
        output_dir.mkdir(exist_ok=True)

        target_images_np = np.array([np.array(img) for img in target_images])
        ref_images_np = np.array([np.array(img) for img in ref_images])
        u_illumination_np = np.array(u_illumination_list)

        np.save(output_dir / "target_intensity_images.npy", target_images_np)
        np.save(output_dir / "ref_intensity_images.npy", ref_images_np)
        np.save(output_dir / "u_illumination_list.npy", u_illumination_np)

        print(f"\nSaved {len(target_images)} intensity images to {output_dir}/")
        print(f"Target images shape: {target_images_np.shape}")
        print(f"Reference images shape: {ref_images_np.shape}")
        print(f"U illumination shape: {u_illumination_np.shape}")

        # Create summary visualization
        _, axes = plt.subplots(2, min(4, num_angles), figsize=(16, 8))
        if num_angles == 1:
            axes = axes.reshape(2, 1)

        for i in range(min(4, num_angles)):
            # Target image
            im1 = axes[0, i].imshow(target_images_np[i], cmap="gray")
            axes[0, i].set_title(f"Target {i + 1} (θ={angles[i]:.2f})")
            plt.colorbar(im1, ax=axes[0, i])

            # Reference image
            im2 = axes[1, i].imshow(ref_images_np[i], cmap="gray")
            axes[1, i].set_title(f"Reference {i + 1}")
            plt.colorbar(im2, ax=axes[1, i])

        plt.tight_layout()
        plt.savefig(output_dir / "intensity_images_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Summary visualization saved to {output_dir}/intensity_images_summary.png")


def main() -> None:
    """Run the main test function."""
    print("Starting intensity image generation test...")

    # Clear JAX cache
    jax.clear_caches()  # type: ignore[no-untyped-call]

    # Setup parameters
    idt_params, mlb_params, _ = setup_test_parameters()

    print(f"IDT parameters: img_size_px={idt_params.img_size_px}, aperturesize_px={idt_params.aperturesize_px}")
    print(f"MLB parameters: xy_shape={mlb_params.xy_shape}, num_layers={mlb_params.num_layers}")

    # Generate simple sphere
    scattering_potential = create_simple_sphere_potential(mlb_params)

    SlicingVisualizer(np.asarray(scattering_potential)).run()

    # Test single image first
    print("\n" + "=" * 50)
    print("TESTING SINGLE INTENSITY IMAGE")
    print("=" * 50)

    test_single_intensity_image(idt_params, mlb_params, scattering_potential, 0.0, True)

    # Test multiple angles
    print("\n" + "=" * 50)
    print("TESTING MULTIPLE INTENSITY IMAGES")
    print("=" * 50)

    test_multiple_angles(idt_params, mlb_params, scattering_potential, 6, True)

    # Clean up
    jax.clear_caches()  # type: ignore[no-untyped-call]
    gc.collect()

    print("\nIntensity image generation test completed!")


if __name__ == "__main__":
    main()
