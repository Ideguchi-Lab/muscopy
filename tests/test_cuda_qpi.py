"""Tests for the optional CuPy QPI implementation."""

from __future__ import annotations

import sys
import types
import typing
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft
from numpy.testing import assert_allclose

from muscopy.dh import MuParameters
from muscopy.dh import offaxis_dh as jax_offaxis_dh
from muscopy.qpi import correct_phase_offset as jax_correct_phase_offset
from muscopy.qpi import mip_qpi as jax_mip_qpi
from muscopy.qpi import qpi as jax_qpi
from muscopy.qpi_utils import unwrap_phase as jax_unwrap_phase

if TYPE_CHECKING:
    from collections.abc import Callable

    import cupy as cp

    AnyFunc: typing.TypeAlias = Callable[..., Any]


class _FakeCupy(types.ModuleType):
    ndarray: type[np.ndarray]
    pi: float
    fft: types.ModuleType
    asarray: AnyFunc
    array: AnyFunc
    asnumpy: AnyFunc
    arange: AnyFunc
    meshgrid: AnyFunc
    ones: AnyFunc
    ones_like: AnyFunc
    exp: AnyFunc
    angle: AnyFunc
    abs: AnyFunc
    mean: AnyFunc
    sum: AnyFunc
    diff: AnyFunc
    where: AnyFunc
    cos: AnyFunc


class _FakeCupyxFft(types.ModuleType):
    dct: AnyFunc
    idct: AnyFunc


def _install_fake_cupy() -> None:
    cupy_module = _FakeCupy("cupy")
    cupy_module.ndarray = np.ndarray
    cupy_module.pi = np.pi
    cupy_module.fft = np.fft
    cupy_module.asarray = np.asarray
    cupy_module.array = np.array
    cupy_module.asnumpy = np.asarray
    cupy_module.arange = np.arange
    cupy_module.meshgrid = np.meshgrid
    cupy_module.ones = np.ones
    cupy_module.ones_like = np.ones_like
    cupy_module.exp = np.exp
    cupy_module.angle = np.angle
    cupy_module.abs = np.abs
    cupy_module.mean = np.mean
    cupy_module.sum = np.sum
    cupy_module.diff = np.diff
    cupy_module.where = np.where
    cupy_module.cos = np.cos

    cupyx_module = types.ModuleType("cupyx")
    cupyx_scipy_module = types.ModuleType("cupyx.scipy")
    cupyx_fft_module = _FakeCupyxFft("cupyx.scipy.fft")
    cupyx_fft_module.dct = scipy.fft.dct
    cupyx_fft_module.idct = scipy.fft.idct

    sys.modules["cupy"] = cupy_module
    sys.modules["cupyx"] = cupyx_module
    sys.modules["cupyx.scipy"] = cupyx_scipy_module
    sys.modules["cupyx.scipy.fft"] = cupyx_fft_module


_install_fake_cupy()

from muscopy.cuda import dh as cupy_dh  # noqa: E402
from muscopy.cuda import qpi as cupy_qpi  # noqa: E402
from muscopy.cuda import qpi_utils as cupy_qpi_utils  # noqa: E402


def _cp_array(array: object) -> cp.ndarray:
    return typing.cast("cp.ndarray", np.asarray(array))


def _np_array(array: cp.ndarray) -> np.ndarray:
    return typing.cast("np.ndarray", array)


def _qpi_fixture() -> tuple[MuParameters, jnp.ndarray, jnp.ndarray, tuple[int, int]]:
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=64, px_size_m=1e-6, n_sol=1.33)
    x, y = jnp.meshgrid(
        jnp.arange(params.img_size_px),
        jnp.arange(params.img_size_px),
        indexing="ij",
    )
    sample = 1.0 + 0.12 * jnp.sin(2 * jnp.pi * x / params.img_size_px)
    reference = 1.0 + 0.08 * jnp.cos(2 * jnp.pi * y / params.img_size_px)
    offaxis_center = params.img_center
    return params, sample, reference, offaxis_center


def test_cuda_imports_with_cupy_available() -> None:
    assert cupy_dh.__all__ == [
        "correct_aberration",
        "crop_array",
        "get_spectrum",
        "get_spectrums",
        "make_disk",
        "offaxis_dh",
    ]
    assert cupy_qpi.__all__ == ["correct_phase_offset", "mip_qpi", "qpi"]
    assert cupy_qpi_utils.__all__ == ["unwrap_phase"]


def test_offaxis_dh_cupy_matches_jax_single_center() -> None:
    params, array, reference, offaxis_center = _qpi_fixture()

    expected = jax_offaxis_dh(array, reference, params, offaxis_center)
    observed = cupy_dh.offaxis_dh(_cp_array(array), _cp_array(reference), params, offaxis_center)

    assert observed.shape == expected.shape
    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-5)


def test_offaxis_dh_cupy_matches_jax_multiple_centers() -> None:
    params, array, reference, offaxis_center = _qpi_fixture()
    offaxis_centers = [offaxis_center, offaxis_center]

    expected = jax_offaxis_dh(array, reference, params, offaxis_centers)
    observed = cupy_dh.offaxis_dh(_cp_array(array), _cp_array(reference), params, offaxis_centers)

    assert isinstance(observed, list)
    assert len(observed) == len(expected)
    for observed_array, expected_array in zip(observed, expected, strict=True):
        assert_allclose(_np_array(observed_array), np.asarray(expected_array), atol=1e-5)


def test_qpi_cuda_matches_jax_single_center() -> None:
    params, array, reference, offaxis_center = _qpi_fixture()

    expected = jax_qpi(array, reference, params, offaxis_center)
    observed = cupy_qpi.qpi(_cp_array(array), _cp_array(reference), params, offaxis_center)

    assert observed.shape == expected.shape
    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-5)


def test_qpi_cuda_matches_jax_multiple_centers() -> None:
    params, array, reference, offaxis_center = _qpi_fixture()
    offaxis_centers = [offaxis_center, offaxis_center]

    expected = jax_qpi(array, reference, params, offaxis_centers)
    observed = cupy_qpi.qpi(_cp_array(array), _cp_array(reference), params, offaxis_centers)

    assert isinstance(observed, list)
    assert len(observed) == len(expected)
    for observed_array, expected_array in zip(observed, expected, strict=True):
        assert_allclose(_np_array(observed_array), np.asarray(expected_array), atol=1e-5)


def test_qpi_cuda_pupil_shape_validation() -> None:
    params, array, reference, offaxis_center = _qpi_fixture()

    with pytest.raises(ValueError, match=r"Spectrum shape .* and pupil function shape .* must match"):
        cupy_qpi.qpi(
            _cp_array(array),
            _cp_array(reference),
            params,
            offaxis_center,
            pupil_func=_cp_array(np.ones((3, 3))),
        )


def test_mip_qpi_cuda_matches_jax() -> None:
    params, array_off, _, offaxis_center = _qpi_fixture()
    array_on = array_off * 1.03

    expected = jax_mip_qpi(array_on, array_off, params, offaxis_center)
    observed = cupy_qpi.mip_qpi(
        _cp_array(array_on),
        _cp_array(array_off),
        params,
        offaxis_center,
    )

    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-5)


def test_mip_qpi_cuda_crop_center_returns_expected_shape() -> None:
    params, array_off, _, offaxis_center = _qpi_fixture()
    array_on = array_off * 1.03

    observed = cupy_qpi.mip_qpi(
        _cp_array(array_on),
        _cp_array(array_off),
        params,
        offaxis_center,
        crop_center=True,
        c_r=3,
    )

    assert observed.shape == (params.aperturesize_px, params.aperturesize_px)
    assert np.all(np.isfinite(_np_array(observed)))


def test_mip_qpi_cuda_shape_mismatch() -> None:
    params = MuParameters(na=0.1, wavelength_m=500e-9, img_size_px=16, px_size_m=1e-6, n_sol=1.33)

    with pytest.raises(ValueError, match="Array on and off must have the same shape"):
        cupy_qpi.mip_qpi(_cp_array(np.ones((16, 16))), _cp_array(np.ones((8, 8))), params, (8, 8))


def test_mip_qpi_cuda_center_region_inversion() -> None:
    params, array_off, _, offaxis_center = _qpi_fixture()
    array_on = array_off * 1.03

    no_region = cupy_qpi.mip_qpi(_cp_array(array_on), _cp_array(array_off), params, offaxis_center)
    with_region = cupy_qpi.mip_qpi(
        _cp_array(array_on),
        _cp_array(array_off),
        params,
        offaxis_center,
        mip_center_reg=((0, 3), (0, 3)),
    )

    assert_allclose(_np_array(with_region), _np_array(no_region), atol=1e-5)


def test_correct_phase_offset_cupy_matches_jax() -> None:
    phase_array = jnp.arange(16, dtype=float).reshape(4, 4)
    offset_regions = [((0, 2), (0, 2)), ((2, 4), (2, 4))]

    expected = jax_correct_phase_offset(phase_array, offset_regions)
    observed = cupy_qpi.correct_phase_offset(_cp_array(phase_array), offset_regions)

    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-7)


def test_unwrap_phase_cupy_matches_jax() -> None:
    n = 32
    x, y = jnp.meshgrid(jnp.linspace(-3, 3, n), jnp.linspace(-3, 3, n), indexing="ij")
    true_phase = 5 * jnp.exp(-(x**2 + y**2) / 2)
    wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))

    expected = jax_unwrap_phase(wrapped_phase, keep_mean=False)
    observed = cupy_qpi_utils.unwrap_phase(_cp_array(wrapped_phase), keep_mean=False)

    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-5)


def test_unwrap_phase_cupy_with_roi_matches_jax() -> None:
    n = 32
    x, y = jnp.meshgrid(jnp.linspace(-2, 2, n), jnp.linspace(-2, 2, n), indexing="ij")
    true_phase = 3 * jnp.exp(-(x**2 + y**2) / 2) + 0.5
    wrapped_phase = jnp.angle(jnp.exp(1j * true_phase))
    xx, yy = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    roi = ((xx - n // 2) ** 2 + (yy - n // 2) ** 2) < (n // 4) ** 2

    expected = jax_unwrap_phase(wrapped_phase, roi=roi, keep_mean=True)
    observed = cupy_qpi_utils.unwrap_phase(_cp_array(wrapped_phase), roi=_cp_array(roi), keep_mean=True)

    assert_allclose(_np_array(observed), np.asarray(expected), atol=1e-5)


def test_unwrap_phase_cupy_skimage_path() -> None:
    n = 16
    phase = _cp_array(np.zeros((n, n)))

    observed = cupy_qpi_utils.unwrap_phase(phase, use_skimage=True)

    assert isinstance(_np_array(observed), np.ndarray)
    assert observed.shape == phase.shape
