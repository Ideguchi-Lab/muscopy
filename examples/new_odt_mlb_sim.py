# --- updated_main.py ---
# This version of `main` takes `n` and `r` as arguments and returns reconstruction error

import jax.numpy as jnp
from muscopy.odt import odt
from examples.odt_with_mlb_simulation import (
    _setup_parameters,
    _generate_holograms,
    _extract_spectra,
    generate_sphere_potential,
)


def run_single_odt_reconstruction(n: float, r: float, gt_volume: jnp.ndarray | None = None) -> float:
    """
    Run ODT reconstruction for a single (n, r) pair and return error.

    Parameters
    ----------
        n: `float`
    Refractive index of sphere
        r: `float`
    Radius in microns
        gt_volume: `Array | None`
    Optional ground truth volume to compute error

    Returns
    -------
        `float`
        MSE between reconstructed and ground truth volume
    """
    # Setup ODT and MLB simulation parameters
    odt_params, mlb_params, precision = _setup_parameters()
    offaxis_center = (200, 200)
    num_angles = 10

    delta_n = n - 1.33
    scattering_potential = generate_sphere_potential(
        mlb_params, r, delta_n, precision=precision
    )

    # Generate holograms and extract spectra
    target_holograms, ref_holograms = _generate_holograms(
        odt_params, mlb_params, scattering_potential, offaxis_center, precision, num_angles
    )
    cp_spectrums, ref_cp_spectrums = _extract_spectra(target_holograms, ref_holograms, odt_params, offaxis_center)

    # Run ODT reconstruction
    odt_config = {
        "approx_type": "Rytov",
        "hermite_symmetry": True,
        "precision": precision,
        "edge_size": 0,
    }
    refractive_index, _ = odt(cp_spectrums, ref_cp_spectrums, odt_params, odt_config)
    n_reconstructed = jnp.real(refractive_index) - odt_params.n_sol

    # If no ground truth provided, generate GT volume here
    if gt_volume is None:
        gt_volume = scattering_potential / (2 * mlb_params.n_medium)  # same scaling as in reconstruction

    # Compute MSE
    error = jnp.mean((n_reconstructed - gt_volume) ** 2)
    return float(error)


# Optional test
if __name__ == "__main__":
    test_n = 1.45
    test_r = 5.0
    mse = run_single_odt_reconstruction(test_n, test_r)
    print(f"Reconstruction MSE for n={test_n}, r={test_r}: {mse:.4e}")