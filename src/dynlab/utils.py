import numpy as np
from typing import Callable
from scipy.integrate import odeint


def odeint_wrapper(
        f: Callable[[float, tuple[float, float]], tuple],
        t: tuple[float, float],
        Y: tuple[float, float],
        **kwargs
) -> np.ndarray[float]:
    """ wrapper function for scipy.integrate.odeint. Used so that it has the same parameter
        order as scipy.integrate.solve_ivp.
        Args:
            f (function): the vector function from which to calculate the FTLE, f must take 2
                arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                the x position and the y position [x, y].
            t (tuple): the time interval over which to calculate FTLE values, t0 to tf.
            Y (tuple): the initial position of the integration.
        Returns:
            (np.ndarray): the trajectory that was integrated over.
    """
    return odeint(
        f, Y, t, tfirst=True, **kwargs
    )


def force_eigenvectors2D(
        Xi: np.ndarray[np.ndarray[np.ndarray[float]]]
) -> np.ndarray[np.ndarray[np.ndarray[float]]]:
    """ Forces eigenvectors to have the orientation.
        Args:
            Xi (np.ndarray): The eigenvector field to force.
        Returns:
            Xi_forced (np.ndarray): The forced orientation eigenvector field.
    """
    ly, lx, lz = Xi.shape
    Xi_forced = np.empty((ly, lx, lz))
    for i in range(ly):
        for j in range(lx):
            if (
                ((abs(Xi[i, j, 0]) >= abs(Xi[i, j, 1])) and Xi[i, j, 0] < 0)
                or ((abs(Xi[i, j, 0]) < abs(Xi[i, j, 1])) and Xi[i, j, 1] < 0)
            ):
                Xi_forced[i, j, :] = -Xi[i, j, :]
            else:
                Xi_forced[i, j, :] = Xi[i, j, :]
    return Xi_forced
