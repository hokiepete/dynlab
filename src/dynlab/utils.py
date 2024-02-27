import numpy as np
from scipy.integrate import odeint


def odeint_wrapper(f, t, Y, **kwargs):
    return odeint(
        f, Y, t, tfirst=True, **kwargs
    )


def force_eigenvectors2D(Xi):
    ly, lx, lz = Xi.shape
    Xi_norm = np.empty((ly, lx, lz))
    for i in range(ly):
        for j in range(lx):
            if (
                ((abs(Xi[i, j, 0]) >= abs(Xi[i, j, 1])) and Xi[i, j, 0] < 0)
                or ((abs(Xi[i, j, 0]) < abs(Xi[i, j, 1])) and Xi[i, j, 1] < 0)
            ):
                Xi_norm[i, j, :] = -Xi[i, j, :]
            else:
                Xi_norm[i, j, :] = Xi[i, j, :]
    return Xi_norm
