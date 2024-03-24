# The following code is translated from original MATLAB code by the ChatGPT

try:
    import cupy as xp

    _cp = True
except:
    import numpy as xp

    _cp = False
from scipy.fftpack import dctn, idctn


def phase_unwrap(J, weight=None):
    if weight is None:  # unweighted phase unwrap
        # get the wrapped differences of the wrapped values
        dx = xp.concatenate(
            (
                xp.zeros((J.shape[0], 1)),
                xp.unwrap(J, axis=1)[:, 1:] - xp.unwrap(J, axis=1)[:, :-1],
                xp.zeros((J.shape[0], 1)),
            ),
            axis=1,
        )
        dy = xp.concatenate(
            (
                xp.zeros((1, J.shape[1])),
                xp.unwrap(J, axis=0)[1:, :] - xp.unwrap(J, axis=0)[:-1, :],
                xp.zeros((1, J.shape[1])),
            ),
            axis=0,
        )
        rho = xp.diff(dx, axis=1) + xp.diff(dy, axis=0)

        # get the result by solving the poisson equation
        phi = solvePoisson(rho)

    else:  # weighted phase unwrap
        # check if the weight has the same size as J
        if not xp.all(weight.shape == J.shape):
            raise ValueError(
                "Size of the weight must be the same as size of the wrapped phase"
            )

        # vector b in the paper (eq 15) is dx and dy
        dx = xp.concatenate(
            (
                xp.unwrap(J, axis=1)[:, 1:] - xp.unwrap(J, axis=1)[:, :-1],
                xp.zeros((J.shape[0], 1)),
            ),
            axis=1,
        )
        dy = xp.concatenate(
            (
                xp.unwrap(J, axis=0)[1:, :] - xp.unwrap(J, axis=0)[:-1, :],
                xp.zeros((1, J.shape[1])),
            ),
            axis=0,
        )

        # multiply the vector b by weight square (W^T * W)
        WW = weight * weight
        WWdx = WW * dx
        WWdy = WW * dy

        # applying A^T to WWdx and WWdy is like obtaining rho in the unweighted case
        WWdx2 = xp.concatenate((xp.zeros((J.shape[0], 1)), WWdx), axis=1)
        WWdy2 = xp.concatenate((xp.zeros((1, J.shape[1])), WWdy), axis=0)
        rk = xp.diff(WWdx2, axis=1) + xp.diff(WWdy2, axis=0)
        normR0 = xp.linalg.norm(rk)

        # start the iteration
        eps = 1e-8
        k = 0
        phi = xp.zeros_like(J)
        while not xp.all(rk == 0):
            zk = solvePoisson(rk)
            k += 1

            if k == 1:
                pk = zk
            else:
                betak = xp.sum(rk * zk) / xp.sum(rkprev * zkprev)
                pk = zk + betak * pk

            # save the current value as the previous values
            rkprev = rk
            zkprev = zk

            # perform one scalar and two vectors update
            Qpk = applyQ(pk, WW)
            alphak = xp.sum(rk * zk) / xp.sum(pk * Qpk)
            phi += alphak * pk
            rk -= alphak * Qpk

            # check the stopping conditions
            if k >= xp.size(J) or xp.linalg.norm(rk) < eps * normR0:
                break

    return phi


def solvePoisson(rho):
    # solve the poisson equation using dct
    dctRho = dctn(rho, type=2)
    N, M = rho.shape
    I, J = xp.meshgrid(xp.arange(M), xp.arange(N))
    dctPhi = dctRho / (2 * (xp.cos(xp.pi * I / M) + xp.cos(xp.pi * J / N) - 2))
    dctPhi[0, 0] = 0  # handling the inf/nan value

    # now invert to get the result
    phi = idctn(dctPhi, type=3)
    return phi


# apply the transformation (A^T)(W^T)(W)(A) to 2D matrix


def applyQ(p, WW):
    # apply (A)
    dx = xp.concatenate((xp.diff(p, axis=1), xp.zeros((p.shape[0], 1))), axis=1)
    dy = xp.concatenate((xp.diff(p, axis=0), xp.zeros((1, p.shape[1]))), axis=0)

    # apply (W^T)(W)
    WWdx = WW * dx
    WWdy = WW * dy

    # apply (A^T)
    WWdx2 = xp.concatenate((xp.zeros((p.shape[0], 1)), WWdx), axis=1)
    WWdy2 = xp.concatenate((xp.zeros((1, p.shape[1])), WWdy), axis=0)
    Qp = xp.diff(WWdx2, axis=1) + xp.diff(WWdy2, axis=0)
    return Qp
