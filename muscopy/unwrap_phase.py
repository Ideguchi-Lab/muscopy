import muscopy.cfg as mcfg

if mcfg._cp:
    import cupy as xp
    from cupyx.scipy.fft import dct as cp_dct
    from cupyx.scipy.fft import idct as cp_idct
else:
    import numpy as xp
    from scipy.fftpack import dct, idct


def dct2(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def idct2(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def cp_dct2(block):
    return cp_dct(cp_dct(block.T, norm="ortho").T, norm="ortho")


def cp_idct2(block):
    return cp_idct(cp_idct(block.T, norm="ortho").T, norm="ortho")


def wraptopi(x):
    xwrap = xp.remainder(x, 2 * xp.pi)
    mask = xp.abs(xwrap) > xp.pi
    xwrap[mask] -= 2 * xp.pi * xp.sign(xwrap[mask])
    mask1 = x < 0
    mask2 = xp.remainder(x, xp.pi) == 0
    mask3 = xp.remainder(x, 2 * xp.pi) != 0
    xwrap[mask1 & mask2 & mask3] -= 2 * xp.pi
    return xwrap


def phase_unwrap(J):
    # get the wrapped differences of the wrapped values
    dx = xp.concatenate(
        (
            xp.zeros((J.shape[0], 1)),
            wraptopi(xp.diff(J, axis=1, n=1)),
            xp.zeros((J.shape[0], 1)),
        ),
        axis=1,
    )
    dy = xp.concatenate(
        (
            xp.zeros((1, J.shape[1])),
            wraptopi(xp.diff(J, axis=0, n=1)),
            xp.zeros((1, J.shape[1])),
        ),
        axis=0,
    )
    rho = xp.diff(dx, axis=1, n=1) + xp.diff(dy, axis=0, n=1)
    # get the result by solving the poisson equation
    phi = solve_poisson(rho)
    return phi


def solve_poisson(rho):
    # solve the Poisson equation using DCT
    if mcfg._cp:
        dctRho = cp_dct2(rho)
    else:
        dctRho = dct2(rho)
    N, M = rho.shape
    I, J = xp.meshgrid(xp.arange(0, M), xp.arange(0, N))
    dctPhi = dctRho / (2 * (xp.cos(xp.pi * I / M) + xp.cos(xp.pi * J / N) - 2))
    dctPhi[0, 0] = 0  # handling the inf/nan value
    if mcfg._cp:
        phi = cp_idct2(dctPhi)
    else:
        phi = idct2(dctPhi)
    return phi
