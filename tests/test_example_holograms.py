"""Regression tests for the inputs used in the holography examples."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from muscopy_example_data.holograms import inline_holograms, main, off_axis_holograms
from numpy.testing import assert_allclose

from muscopy.dh import MuParameters, inline_dh
from muscopy.qpi import qpi

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_off_axis_holograms_reconstruct_disk_phase() -> None:
    params = MuParameters(na=0.8, wavelength_m=500e-9, img_size_px=128, px_size_m=5e-6 / 60, n_sol=1.33)
    off_axis_center = (23, 27)

    hologram, reference = off_axis_holograms(params, off_axis_center)
    phase = qpi(hologram, reference, params, off_axis_center)

    center = params.aperturesize_px // 2
    assert phase.shape == (params.aperturesize_px, params.aperturesize_px)
    assert_allclose(phase[center, center], 2 * np.pi / 3, atol=0.05)
    assert_allclose(phase[0, 0], 0, atol=0.05)


def test_multiplexed_holograms_mix_fields_before_recording_intensity() -> None:
    params = MuParameters(na=0.8, wavelength_m=500e-9, img_size_px=64, px_size_m=5e-6 / 60, n_sol=1.33)
    field = jnp.ones((64, 64))
    fields = jnp.stack((field, 1j * field))
    mixing = jnp.array([[0.5, 0.5], [-0.5, 0.5]])

    holograms, reference = off_axis_holograms(params, (10, 15), fields, multiplex_matrix=mixing)

    assert holograms.shape == (2, 64, 64)
    assert reference.shape == (64, 64)
    # At the origin the reference wave has zero phase and each field has amplitude 1/64.
    assert_allclose(holograms[:, 0, 0], np.array([2.5, 0.5]) / 64**2, rtol=1e-6)
    assert_allclose(reference[0, 0], 4 / 64**2, rtol=1e-6)


def test_inline_holograms_reconstruct_object_field() -> None:
    params = MuParameters(na=0.6, wavelength_m=633e-9, img_size_px=128, px_size_m=5.2e-6, n_sol=1.0)
    phases = jnp.linspace(0, 2 * jnp.pi, 4, endpoint=False)
    ref_amplitude = 0.75

    holograms, expected_field = inline_holograms(params, phases, ref_amplitude)
    reconstructed_field = inline_dh(holograms, phases, params, ref_amp=ref_amplitude)

    assert holograms.shape == (4, 128, 128)
    assert_allclose(reconstructed_field, expected_field, atol=1e-6)


def test_generate_hologram_command_writes_reconstruction_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "holograms.npz"
    monkeypatch.setattr(sys, "argv", ["muscopy-generate-hologram", str(output)])

    main()

    with np.load(output) as data:
        assert data["hologram"].shape == data["reference_hologram"].shape == (512, 512)
        assert data["hologram"].dtype == data["reference_hologram"].dtype == np.float32
        assert_allclose(data["off_axis_center"], (100, 100))
        assert_allclose(data["wavelength_m"], 500e-9)
        assert set(data.files) == {
            "hologram",
            "reference_hologram",
            "off_axis_center",
            "na",
            "wavelength_m",
            "img_size_px",
            "px_size_m",
            "n_sol",
        }
