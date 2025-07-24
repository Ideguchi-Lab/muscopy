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
- muscopy_mlbsim package must be installed
- GPU support (JAX) is recommended for faster computation
"""

# pyright: reportPossiblyUnboundVariable=false, reportInvalidTypeForm=false

import typing

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import Array
from tqdm import tqdm

from muscopy.cfg import ArrayPrecision
from muscopy.dh import get_spectrum
from muscopy.odt import ODTConfig, ODTParameters, odt

try:
    from muscopy_mlbsim import (  # pyright: ignore[reportMissingImports]
        HologramGenerator,
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
    HologramGenerator: typing.Any = None  # type: ignore[no-redef]
    MLBForward: typing.Any = None  # type: ignore[no-redef]
    MLBParameters: typing.Any = None  # type: ignore[no-redef]
    get_oblique_wave_fft: typing.Any = None  # type: ignore[no-redef]
    get_scatter_potential: typing.Any = None  # type: ignore[no-redef]


class HologramSetGenerator:
    """Generate hologram sets using MLB simulation and muscopy_mlbsim.HologramGenerator."""

    def __init__(
        self,
        odt_params: ODTParameters,
        mlb_params: MLBParameters,
        offaxis_center: tuple[int, int],
        precision: ArrayPrecision,
    ) -> None:
        if not MLB_AVAILABLE:
            msg = "muscopy_mlbsim is required but not available"
            raise ImportError(msg)

        self.odt_params = odt_params
        self.mlb_params = mlb_params
        self.precision = precision
        self.offaxis_center = offaxis_center

        # Initialize MLB forward simulator and hologram generator
        self.mlb_forward = MLBForward(mlb_params)
        self.hologram_generator = HologramGenerator(mlb_params)

        # Configure off-axis position
        self.hologram_generator.set_offaxis_position(offaxis_center[0], offaxis_center[1])

        # Store illumination angles
        self.angles: np.ndarray[typing.Any, np.dtype[np.floating[typing.Any]]] | None = None

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

    def generate_hologram_set(
        self,
        amplitude_ref: float = 1.0,
    ) -> tuple[list[Array], list[Array]]:
        r"""Generate hologram set with different illumination angles.

        Returns
        -------
        tuple[list[Array], list[Array]]
            Target holograms and reference holograms

        Raises
        ------
        ValueError
            If illumination angles are not set
        """
        if self.angles is None:
            msg = "Illumination angles not set. Call set_illumination_angles() first."
            raise ValueError(msg)

        target_holograms = []
        reference_holograms = []
        hologram_shape = (self.odt_params.img_size_px, self.odt_params.img_size_px)

        print("Generating holograms...")
        for angle in tqdm(self.angles):
            # Calculate illumination wave vector components
            kx_ill = (
                self.odt_params.light_freq_px
                * self.odt_params.na_illumination
                / self.odt_params.n_sol
                * self.odt_params.k_per_px
                * np.cos(angle)
            )
            ky_ill = (
                self.odt_params.light_freq_px
                * self.odt_params.na_illumination
                / self.odt_params.n_sol
                * self.odt_params.k_per_px
                * np.sin(angle)
            )

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
                hologram_shape=hologram_shape,
                reference_amplitude=amplitude_ref,
                bit_depth=16,
                output_dtype="float32",  # Keep as float for processing
            )
            target_holograms.append(hologram)

            # Generate reference hologram (no scattering)
            ref_field = jnp.fft.ifft2(jnp.fft.ifftshift(input_field_fft))
            self.hologram_generator.set_target_field(ref_field)
            ref_hologram = self.hologram_generator.generate_hologram(
                hologram_shape=hologram_shape,
                reference_amplitude=amplitude_ref,
                bit_depth=16,
                output_dtype="float32",
            )
            reference_holograms.append(ref_hologram)

        return target_holograms, reference_holograms


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


def _setup_parameters() -> tuple[ODTParameters, MLBParameters, ArrayPrecision]:
    r"""Set up ODT and MLB simulation parameters.

    Returns
    -------
    tuple[ODTParameters, MLBParameters, ArrayPrecision]
        ODT parameters, MLB parameters, and array precision settings
    """
    # Use 32-bit precision to avoid JAX complex128 warnings (complex64 is sufficient)
    precision = ArrayPrecision(int_length=16, float_length=32)

    # ODT parameters
    print("Setting ODT parameters...")
    odt_params = ODTParameters(
        na=1.1,
        wavelength_m=532e-9,
        img_size_px=512,  # Smaller for faster computation
        px_size_m=3.45e-6 * 3 / 180 / 2,
        n_sol=1.33,
        na_illumination=1.0,
    )

    # MLB simulation parameters
    print("Setting MLB simulation parameters...")
    mlb_params = MLBParameters(
        wavelength_m=odt_params.wavelength_m,
        numerical_aperture=odt_params.na,
        n_medium=odt_params.n_sol,
        xy_shape=(2 * odt_params.aperturesize_px, 2 * odt_params.aperturesize_px),
        num_layers=odt_params.freq_axial_extent_px,
        dxy_m=odt_params.imgpx_lateral_m_per_px,
        dz_m=odt_params.imgpx_axial_m_per_px,
    )

    return odt_params, mlb_params, precision


def _generate_holograms(  # noqa: PLR0913, PLR0917
    odt_params: ODTParameters,
    mlb_params: MLBParameters,
    scattering_potential: Array,
    offaxis_center: tuple[int, int],
    precision: ArrayPrecision,
    num_angles: int = 60,
) -> tuple[list[Array], list[Array]]:
    r"""Generate hologram sets.

    Returns
    -------
    tuple[list[Array], list[Array]]
        Target holograms and reference holograms
    """
    print("Setting up hologram generator...")
    hol_gen = HologramSetGenerator(odt_params, mlb_params, offaxis_center, precision)
    hol_gen.set_illumination_angles(num_angles)  # Fewer angles for faster computation
    hol_gen.set_scattering_potential(scattering_potential)

    return hol_gen.generate_hologram_set()


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


def main() -> None:  # noqa: PLR0914
    """Demonstrate ODT with MLB simulation."""
    if not MLB_AVAILABLE:
        print("Skipping ODT with MLB simulation demo - muscopy_mlbsim not available.")
        # Create a simple placeholder plot for documentation
        _, ax = plt.subplots(figsize=(8, 6))
        message = (
            "muscopy_mlbsim Required\n\n"
            "This example requires the muscopy_mlbsim package.\n"
            "Please install it with:\n"
            "pip install -e ./muscopy-mlbsim"
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
        plt.title("ODT with MLB Simulation Example")
        plt.tight_layout()
        plt.savefig("odt_mlb_simulation_placeholder.png", dpi=150, bbox_inches="tight")
        plt.show()
        return

    # Setup parameters
    odt_params, mlb_params, precision = _setup_parameters()
    offaxis_center = (200, 200)
    num_angles = 10  # Number of illumination angles for tomographic acquisition

    # Generate sample (sphere)
    print("Generating spherical sample...")
    radius_um = 2.0
    delta_n = 0.02
    
    
    scattering_potential = generate_sphere_potential(mlb_params, radius_um, delta_n, precision=precision)

    print(f"Sample size: {scattering_potential.shape}")
    print(f"Memory usage: {scattering_potential.nbytes / 1024**2:.1f} MB")

    # Generate holograms
    target_holograms, ref_holograms = _generate_holograms(
        odt_params, mlb_params, scattering_potential, offaxis_center, precision, num_angles
    )

    # Extract complex field spectra
    cp_spectrums, ref_cp_spectrums = _extract_spectra(target_holograms, ref_holograms, odt_params, offaxis_center)

    # ODT reconstruction
    print("Performing ODT reconstruction...")
    odt_config = ODTConfig(
        approx_type="Rytov",
        hermite_symmetry=True,
        precision=precision,
        edge_size=0,
    )

    refractive_index, synthetic_spectra = odt(cp_spectrums, ref_cp_spectrums, odt_params, odt_config)

    # Convert to real refractive index
    n_reconstructed = jnp.real(refractive_index) - odt_params.n_sol

    # Visualization
    _visualize_results(n_reconstructed, target_holograms, delta_n)

    # Visualize synthetic spectra profiles
    refractive_index, synthetic_spectra = odt(cp_spectrums, ref_cp_spectrums, odt_params, odt_config)
    visualize_synthetic_spectra_profiles(synthetic_spectra)

    #for access in odtcomparison.py
    return(n_reconstructed)
n_recon = main()



if __name__ == "__main__":
    main()
