from typing import Callable
from itertools import product

import numpy as np

from dynlab.utils import odeint_wrapper
from dynlab.diagnostics._base_classes import LagrangianDiagnostic2D, RidgeExtractor2D


class FTLE(LagrangianDiagnostic2D):
    """ Calculates and stores the FTLE field for a 2 dimensional flow."""
    def __init__(self, integrator: Callable = odeint_wrapper, num_threads: int = 1) -> None:
        super().__init__(integrator=integrator, num_threads=num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        f: Callable[[float, tuple[float, float]], tuple],
        t: tuple[float, float],
        edge_order: int = 1,
        **kwargs
    ) -> np.ndarray[np.ndarray[float, ...], ...]:
        """ Computes the FTLE field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                f (function): the vector function from which to calculate the FTLE, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (tuple): the time interval over which to calculate FTLE values, t0 to tf.
                NOTE: function also accepts kwargs for scipy.integrate.solve_ivp.
            Returns:
                ftle (np.ndarray): The FTLE field for the given flow.
        """

        # computes the flow map
        super().compute(x, y, f, t, **kwargs)

        # Calculate flow map gradients
        dfxdy, dfxdx = np.gradient(
            self.flow_map[:, :, 0].squeeze(), self.y, self.x, edge_order=edge_order
        )
        dfydy, dfydx = np.gradient(
            self.flow_map[:, :, 1].squeeze(), self.y, self.x, edge_order=edge_order
        )

        # initialize FTLE matrix
        self.ftle = np.ma.empty([self.ydim, self.xdim])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dfxdx[i, j], dfxdy[i, j], dfydx[i, j], dfydy[i, j]):
                # Calculate Cauchy-Green tensor, C
                JF = np.array([[dfxdx[i, j], dfxdy[i, j]], [dfydx[i, j], dfydy[i, j]]])
                C = np.dot(JF.T, JF)

                # Calculate FTLE
                lambda_max = np.max(np.linalg.eig(C)[0])
                if lambda_max >= 1:
                    self.ftle[i, j] = 1.0 / (2.0*abs(t[-1] - t[0]))*np.log(lambda_max)
                else:
                    self.ftle[i, j] = 0
            else:
                # If the data is masked, then mask the grid point in the output.
                self.ftle[i, j] = np.ma.masked

        return self.ftle


class LCS(LagrangianDiagnostic2D, RidgeExtractor2D):
    def __init__(self,  integrator: Callable = odeint_wrapper, num_threads: int = 1) -> None:
        super().__init__(integrator=integrator, num_threads=num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        f: Callable[[float, tuple[float, float]], tuple],
        t: tuple[float, float],
        edge_order: int = 1,
        percentile: float = None,
        force_eigenvectors: bool = False,
        debug: bool = False,
        **kwargs
    ) -> np.ndarray[np.ndarray[float, ...], ...]:
        """ 
            NOTE: Numpy's eigenvector algorithm returns unique eigenvectors up to their sign, i.e.
            Eigenvector[0]=Xi_0. Eigenvector[1]=-Xi_1. This sudden change of sign in the
            eigenvector field can lead to a sudden change of sign in the directional derivative
            field and thus the detection of numerical artifactions as coherent structures. If you
            have doubts or concerns about your resutls set debug=True and you can examine the
            Xi_max, directional_derivative, and concavity attributes to determine if the structures
            are numerical artifacts or not.
        
        
        """
        super().compute(x, y, f, t, **kwargs)

        # Calculate flow map gradients
        dfxdy, dfxdx = np.gradient(
            self.flow_map[:, :, 0].squeeze(), self.y, self.x, edge_order=edge_order
        )
        dfydy, dfydx = np.gradient(
            self.flow_map[:, :, 1].squeeze(), self.y, self.x, edge_order=edge_order
        )

        # flow map is no longer needed and some of the downstream calculations can be quite large
        # deleting to be more space efficent and allow larger fields.
        del self.flow_map

        # initialize FTLE and max eigenvector matrices
        self.ftle = np.ma.empty([self.ydim, self.xdim])
        self.Xi_max = np.ma.empty([self.ydim, self.xdim, 2])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dfxdx[i, j], dfxdy[i, j], dfydx[i, j], dfydy[i, j]):
                # Calculate Cauchy-Green tensor, C
                JF = np.array([[dfxdx[i, j], dfxdy[i, j]], [dfydx[i, j], dfydy[i, j]]])
                C = np.dot(JF.T, JF)

                # Calculate FTLE and directional derivative
                lambda_max = np.max([0])
                eigenValues, eigenVectors = np.linalg.eig(C)
                idx = eigenValues.argsort()
                lambda_max = eigenValues[idx[-1]]
                self.Xi_max[i, j, :] = eigenVectors[:, idx[-1]]
                if lambda_max >= 1:
                    self.ftle[i, j] = 1.0 / (2.0*abs(t[-1] - t[0]))*np.log(lambda_max)
                else:
                    self.ftle[i, j] = 0
            else:
                # If the data is masked, then mask the grid point in the output.
                self.ftle[i, j] = np.ma.masked
                self.Xi_max[i, j, 0] = np.ma.masked
                self.Xi_max[i, j, 1] = np.ma.masked

        # derivatives are no longer needed deleting to be more space efficent and allow larger
        # fields.
        del dfxdx, dfxdy, dfydx, dfydy

        if force_eigenvectors:
            self.Xi_max = force_eigenvectors2D(self.Xi_max)

        self.lcs = self.extract_ridges(self.ftle, self.Xi_max, percentile, edge_order, debug)

        # free up some extra memory if not needed
        if not debug:
            del self.Xi_max

        return self.lcs