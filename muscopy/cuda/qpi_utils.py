"""CuPy QPI utility functions."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.fft as cupyx_fft
from skimage.restoration import unwrap_phase as skimage_unwrap_phase

__all__ = ["unwrap_phase"]


def unwrap_phase(
    phase_image: cp.ndarray,
    *,
    roi: cp.ndarray | None = None,
    keep_mean: bool = True,
    use_skimage: bool = False,
) -> cp.ndarray:
    """Unwrap a 2D CuPy phase image using the Poisson solver.

    Parameters
    ----------
    phase_image : `cupy.ndarray`
        The wrapped phase image to be unwrapped.
    roi : `cupy.ndarray`, optional
        A region of interest mask where the unwrapping should be applied. If `None`,
        the entire image is considered. Default is `None`.
    keep_mean : `bool`, optional
        If `True`, the mean of the original phase image is added back to the unwrapped phase.
        Default is `True`.
    use_skimage : `bool`, optional
        If `True`, uses ``skimage.restoration.unwrap_phase`` on CPU and returns
        the result as a CuPy array. If `False`, uses the CuPy Poisson solver method.
        Default is `False`.

    Returns
    -------
    `cupy.ndarray`
        The unwrapped phase image.
    """
    if use_skimage:
        phase_cpu = cp.asnumpy(phase_image)
        unwrapped_cpu = skimage_unwrap_phase(phase_cpu)  # type: ignore[no-untyped-call]
        return cp.asarray(unwrapped_cpu)

    original_roi = roi
    active_roi = cp.ones(phase_image.shape, dtype=bool) if roi is None else roi.astype(bool)

    dx = _wraptopi(cp.diff(phase_image, axis=1))
    dy = _wraptopi(cp.diff(phase_image, axis=0))

    roi_x = active_roi[:, 1:] & active_roi[:, :-1]
    roi_y = active_roi[1:, :] & active_roi[:-1, :]

    dx = cp.where(roi_x, dx, 0.0)
    dy = cp.where(roi_y, dy, 0.0)

    rho = cp.diff(dx, axis=1, prepend=0.0, append=0.0) + cp.diff(dy, axis=0, prepend=0.0, append=0.0)
    rho = cp.where(active_roi, rho, 0.0)

    phi = _solve_poisson(rho)
    phi = cp.where(active_roi, phi, phase_image)

    if keep_mean:
        if original_roi is not None:
            mean_roi = cp.sum(cp.where(active_roi, phase_image, 0.0)) / cp.sum(active_roi)
            phi = cp.where(active_roi, phi + mean_roi, phi)
        else:
            phi += phase_image.mean()

    return phi


def _wraptopi(x: cp.ndarray) -> cp.ndarray:
    return (x + cp.pi) % (2.0 * cp.pi) - cp.pi


def _solve_poisson(rho: cp.ndarray) -> cp.ndarray:
    dct_rho = _dct2(rho)
    n, m = rho.shape
    i, j = cp.meshgrid(cp.arange(0, n), cp.arange(0, m), indexing="ij")

    denom = 2.0 * (cp.cos(cp.pi * i / n) + cp.cos(cp.pi * j / m) - 2.0)
    denom_safe = cp.where((i == 0) & (j == 0), 1.0, denom)

    dct_phi = dct_rho / denom_safe
    dct_phi = dct_phi.copy()
    dct_phi[0, 0] = 0.0
    return _idct2(dct_phi)


def _dct2(x: cp.ndarray) -> cp.ndarray:
    return cupyx_fft.dct(cupyx_fft.dct(x.T, type=2, norm="ortho").T, type=2, norm="ortho")


def _idct2(x: cp.ndarray) -> cp.ndarray:
    return cupyx_fft.idct(cupyx_fft.idct(x.T, type=2, norm="ortho").T, type=2, norm="ortho")
