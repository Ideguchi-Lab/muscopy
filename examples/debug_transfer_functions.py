"""Debug and visualize IDT transfer functions.

This script provides tools to visualize and analyze transfer functions
used in Intensity Diffraction Tomography (IDT) reconstruction.
"""

import matplotlib.pyplot as plt
import numpy as np

from muscopy.idt import IDTParameters, transfer_func_im, transfer_func_re


def visualize_transfer_functions(  # noqa: PLR0914
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
    n_angles_to_show = min(2, len(u_illumination_list))  # Show only 2 angles for better visibility
    angles_to_show = u_illumination_list[:n_angles_to_show]

    # Create figure with subplots (2 rows x 4 columns: |H_re|, |H_im|, phase(H_re), phase(H_im) for each angle)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    # Adjust subplot spacing
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, (u_x, u_y) in enumerate(angles_to_show):
        # Calculate transfer functions
        h_re = transfer_func_re(idt_params, (u_x, u_y), z_slice, 1.0)
        h_im = transfer_func_im(idt_params, (u_x, u_y), z_slice, 1.0)

        # Convert to numpy for plotting
        h_re_np = np.array(h_re) if hasattr(h_re, "__array__") else h_re
        h_im_np = np.array(h_im) if hasattr(h_im, "__array__") else h_im

        # Row 0: First angle (i=0), Row 1: Second angle (i=1)
        row = i
        # Column 0: |H_re| (Magnitude of real transfer function)
        im_mag_re = axes[row, 0].imshow(np.abs(h_re_np), cmap="viridis", aspect="equal")
        axes[row, 0].set_title(f"|H_re|\nAngle: ({u_x:.3f}, {u_y:.3f})", fontsize=11)
        axes[row, 0].set_xlabel("kx [px]", fontsize=9)
        axes[row, 0].set_ylabel("ky [px]", fontsize=9)
        plt.colorbar(im_mag_re, ax=axes[row, 0])

        # Column 1: |H_im| (Magnitude of imaginary transfer function)
        im_mag_im = axes[row, 1].imshow(np.abs(h_im_np), cmap="plasma", aspect="equal")
        axes[row, 1].set_title(f"|H_im|\nAngle: ({u_x:.3f}, {u_y:.3f})", fontsize=11)
        axes[row, 1].set_xlabel("kx [px]", fontsize=9)
        axes[row, 1].set_ylabel("ky [px]", fontsize=9)
        plt.colorbar(im_mag_im, ax=axes[row, 1])

        # Column 2: Phase(H_re) (Phase of real transfer function)
        phase_re = np.angle(h_re_np)
        im_phase_re = axes[row, 2].imshow(phase_re, cmap="hsv", aspect="equal", vmin=-np.pi, vmax=np.pi)
        axes[row, 2].set_title("Phase(H_re)", fontsize=11)
        axes[row, 2].set_xlabel("kx [px]", fontsize=9)
        axes[row, 2].set_ylabel("ky [px]", fontsize=9)
        cbar_re = plt.colorbar(im_phase_re, ax=axes[row, 2])
        cbar_re.set_label("Phase [rad]", fontsize=9)

        # Column 3: Phase(H_im) (Phase of imaginary transfer function)
        phase_im = np.angle(h_im_np)
        im_phase_im = axes[row, 3].imshow(phase_im, cmap="hsv", aspect="equal", vmin=-np.pi, vmax=np.pi)
        axes[row, 3].set_title("Phase(H_im)", fontsize=11)
        axes[row, 3].set_xlabel("kx [px]", fontsize=9)
        axes[row, 3].set_ylabel("ky [px]", fontsize=9)
        cbar_im = plt.colorbar(im_phase_im, ax=axes[row, 3])
        cbar_im.set_label("Phase [rad]", fontsize=9)

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
        z_values = [0.0, 1e-6, 2e-6, 5e-6, 10e-6]  # Default z positions in meters

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


def main() -> None:
    """Run transfer function debugging visualization."""
    # Example usage
    print("Transfer Function Debugging Tool")
    print("================================\n")

    # Set up example IDT parameters
    idt_params = IDTParameters(
        na=0.6,
        wavelength_m=532e-9,
        img_size_px=1024,
        px_size_m=3.45e-6 * 3 / 180,
        n_sol=1.33,
        na_illumination=0.6,
        num_z_slices=256,
    )

    # Generate example illumination angles
    num_angles = 12
    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    u_illumination_list = []

    for angle in angles:
        kx_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.cos(angle)
        ky_ill = idt_params.light_freq_px * idt_params.na_illumination / idt_params.n_sol * np.sin(angle)
        u_illumination_list.append((kx_ill, ky_ill))

    # Analyze transfer function properties
    print("\nAnalyzing transfer function properties...")
    analyze_transfer_function_properties(idt_params, u_illumination_list)

    # Visualize transfer functions at different z positions
    z_positions = [0.0, 2e-6, 5e-6]  # z positions in meters
    for z_pos in z_positions:
        print(f"\nVisualizing transfer functions at z={z_pos * 1e6:.1f}μm...")
        visualize_transfer_functions(
            idt_params, u_illumination_list, z_slice=z_pos, save_path=f"transfer_functions_z{z_pos * 1e6:.1f}um.png"
        )


if __name__ == "__main__":
    main()
