"""IDT with MLB simulation example.

This example demonstrates how to use Intensity Diffraction Tomography (IDT)
with Multi-layer Born (MLB) simulation for microscopy analysis.
"""

# pyright: reportPossiblyUnboundVariable=false, reportInvalidTypeForm=false

import typing

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
from muscopy.idt import IDTParameters, compute_idt


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
            reference_intensity_images.append(ref_hologram)

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


def _setup_parameters() -> tuple[IDTParameters, MLBParameters, ArrayPrecision]:
    r"""Set up ODT and MLB simulation parameters.

    Returns
    -------
    tuple[IDTParameters, MLBParameters, ArrayPrecision]
        ODT parameters, MLB parameters, and array precision settings
    """
    # Use 32-bit precision to avoid JAX complex128 warnings (complex64 is sufficient)
    precision = ArrayPrecision(int_length=16, float_length=32)

    # IDT parameters
    print("Setting ODT parameters...")
    idt_params = IDTParameters(
        na=0.6,
        wavelength_m=532e-9,  # 532 nm
        img_size_px=512,
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.6,
        num_z_slices=256,
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
) -> tuple[tuple[list[Array], list[Array]], list[tuple[float, float]]]:
    r"""Generate hologram sets.

    Returns
    -------
    tuple[tuple[list[Array], list[Array]], list[tuple[float, float]]]
        Target holograms, reference holograms, and illumination angles
    """
    print("Setting up hologram generator...")
    intensity_image_gen = IntensityImageSetGenerator(idt_params, mlb_params, precision)
    intensity_image_gen.set_illumination_angles(num_angles)  # Fewer angles for faster computation
    intensity_image_gen.set_scattering_potential(scattering_potential)

    intensity_images = intensity_image_gen.generate_intensity_image_set()

    u_illumination_list = intensity_image_gen.u_illumination_list
    if u_illumination_list is None:
        u_illumination_list = []

    return intensity_images, u_illumination_list


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

    # Print summary
    print("\nReconstruction Summary:")
    print(f"Shape: {n_reconstructed_np.shape}")
    print(f"Δn range: [{n_reconstructed_np.min():.4f}, {n_reconstructed_np.max():.4f}]")
    print(f"Expected Δn: {delta_n:.4f}")
    print(f"Peak Δn: {n_reconstructed_np.max():.4f}")
    print(f"Recovery ratio: {n_reconstructed_np.max() / delta_n:.2f}")


def main() -> None:
    """Demonstrate IDT with MLB simulation."""
    # Setup parameters
    idt_params, mlb_params, precision = _setup_parameters()
    num_angles = 20  # Number of illumination angles for tomographic acquisition

    # Generate sample (sphere)
    print("Generating spherical sample...")
    radius_um = 2.0
    delta_n = 0.02
    scattering_potential = generate_sphere_potential(mlb_params, radius_um, delta_n, precision=precision)

    print(f"Sample size: {scattering_potential.shape}")
    print(f"Memory usage: {scattering_potential.nbytes / 1024**2:.1f} MB")

    # Generate holograms
    (target_intensity_images, ref_intensity_images), u_illumination_list = _generate_intensity_images(
        idt_params, mlb_params, scattering_potential, precision, num_angles
    )

    n_re, _ = compute_idt(idt_params, target_intensity_images, ref_intensity_images, u_illumination_list)

    # Convert to real refractive index
    n_reconstructed = n_re - idt_params.n_sol

    # Visualization
    _visualize_results(n_reconstructed, target_intensity_images, delta_n)

    # visualize_synthetic_spectra_profiles()


if __name__ == "__main__":
    main()
