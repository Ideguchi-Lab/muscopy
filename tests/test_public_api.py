"""Tests for the v1.0 public API boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

from muscopy import cfg, dh, dir_parser, idt, image_checker, odt, phase_noise, phasor, qpi, qpi_utils

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


def _assert_public_api(module: ModuleType, expected: Sequence[str]) -> None:
    assert module.__all__ == expected
    for name in expected:
        assert hasattr(module, name)


def test_v1_public_api_exports() -> None:
    _assert_public_api(
        cfg,
        [
            "ArrayPrecision",
            "MIPRegion",
            "OffsetRegions",
            "Region",
            "Regions",
        ],
    )
    _assert_public_api(
        dh,
        [
            "MuParameters",
            "correct_aberration",
            "correct_gradient",
            "correct_offset",
            "crop_array",
            "demultiplex_cp_arrays",
            "get_spectrum",
            "get_spectrums",
            "inline_dh",
            "make_disk",
            "offaxis_dh",
            "print_all_parameters",
            "ps_idh_reconstruct",
        ],
    )
    _assert_public_api(qpi, ["correct_phase_offset", "mip_qpi", "qpi"])
    _assert_public_api(qpi_utils, ["unwrap_phase"])
    _assert_public_api(
        odt,
        [
            "EwaldEmbeddingMode",
            "ODTConfig",
            "ODTParameters",
            "ScatteringSpectrum",
            "calc_scattering_potential_from_spectrums",
            "calc_scattering_spectrums",
            "calculate_odt_difference",
            "odt",
        ],
    )
    _assert_public_api(
        idt,
        [
            "IDTConfig",
            "IDTParameters",
            "IDTZCenteringMode",
            "compute_g_list",
            "compute_idt",
            "compute_permittivity",
            "convert_to_refractive_index",
            "fourier_transform",
            "transfer_func_im",
            "transfer_func_re",
        ],
    )
    _assert_public_api(phasor, ["PhasorParameters", "PhasorResult", "phasor"])
    _assert_public_api(phase_noise, ["calc_phase_noise", "calc_visibility"])
    _assert_public_api(
        dir_parser,
        [
            "numpy_parser",
            "png_parser",
            "recursive_file_parser",
            "recursive_numpy_parser",
            "tiff_parser",
        ],
    )


def test_pre_v1_legacy_api_names_are_removed() -> None:
    assert not hasattr(odt, "calc_scattering_potential")
    assert not hasattr(idt, "compute_permitivity")


def test_idt_and_odt_helpers_are_outside_stable_public_api() -> None:
    helper_names_by_module = {
        idt: [
            "_idt_coherent_pupil_radius_px",
            "_idt_intensity_support_radius_px",
            "_make_frequency_grid_xy",
            "_make_green_func",
            "_make_pupil_func",
            "_make_z_position",
            "_relative_imag_residual",
            "_validate_idt_params",
            "_validate_idt_sampling",
            "_validate_xy_image",
        ],
        odt: [
            "_calc_refractive_index",
            "_discard_higher_axial_freq",
            "_fill_hermite_components",
            "_synthesize_spectrum",
            "_zeropad_higher_axial_freq",
        ],
    }

    for module, helper_names in helper_names_by_module.items():
        for helper_name in helper_names:
            assert hasattr(module, helper_name)
            assert helper_name not in module.__all__


def test_experimental_image_checker_has_no_stable_export_contract() -> None:
    assert not hasattr(image_checker, "__all__")
