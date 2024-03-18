import numpy as np
from typing import Callable
from itertools import product

from dynlab.diagnostics._base_classes import EulerianDiagnostic2D, RidgeExtractor2D
from dynlab.utils import force_eigenvectors2D


class _AttractionRepulsionRate(EulerianDiagnostic2D):
    """ Computes and stores the attraction and repulsion rate fields (s_1 & s_n) for a
            2 dimensional flow.
    """
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...] = None,
        v: np.ndarray[np.ndarray[float, ...], ...] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Computes the attraction rate and repulsion rate fields for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
            Returns:
                attraction_rate (np.ndarray): The attraction rate field.
                repulsion_rate (np.ndarray): The repulsion rate field.
        """
        super().compute(x, y, u, v, f, t)

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.attraction_rate = np.ma.empty([self.ydim, self.xdim])
        self.repulsion_rate = np.ma.empty([self.ydim, self.xdim])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                eigenValues, _ = np.linalg.eig(S)
                idx = eigenValues.argsort()
                self.attraction_rate[i, j] = eigenValues[idx[0]]
                self.repulsion_rate[i, j] = eigenValues[idx[-1]]
            else:
                # If the data is masked, then mask the grid point in the output.
                self.attraction_rate[i, j] = np.ma.masked
                self.repulsion_rate[i, j] = np.ma.masked
        return self.attraction_rate, self.repulsion_rate


class AttractionRate(_AttractionRepulsionRate)
    def compute(
        self,
        x: np.ndarray[float, np.Any],
        y: np.ndarray[float, np.Any],
        u: np.ndarray[np.ndarray[float, np.Any], np.Any] = None,
        v: np.ndarray[np.ndarray[float, np.Any], np.Any] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[ndarray, ndarray]:
        self.attraction_rate = super().compute(x, y, u, v, f, t, edge_order)
        return self.attraction_rate


class RepulsionRate(_AttractionRepulsionRate)
    def compute(
        self,
        x: np.ndarray[float, np.Any],
        y: np.ndarray[float, np.Any],
        u: np.ndarray[np.ndarray[float, np.Any], np.Any] = None,
        v: np.ndarray[np.ndarray[float, np.Any], np.Any] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[ndarray, ndarray]:
        self.repulsion_rate = super().compute(x, y, u, v, f, t, edge_order)
        return self.repulsion_rate


class TrajectoryRepulsionRate(EulerianDiagnostic2D):
    """ Computes and stores the rhodot field for a 2 dimensional flow.
    """
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...] = None,
        v: np.ndarray[np.ndarray[float, ...], ...] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Computes the rhodot field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
            Returns:
                rhodot (np.ndarray): The rhodot field.
        """
        super().compute(x, y, u, v, f, t)

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.rhodot = np.ma.empty([self.ydim, self.xdim])
        J = np.array([[0, 1], [-1, 0]])
        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                Velocity = np.array([self.u[i, j], self.v[i, j]])
                Velocity_Squared = np.dot(Velocity, Velocity)
                if Velocity_Squared:
                    self.rhodot[i, j] = np.dot(
                        Velocity, np.dot(np.matmul(J.T, np.matmul(S, J)), Velocity)
                    ) / Velocity_Squared
                else:
                    # If V dot V = 0, then mask the grid point in the output.
                    self.rhodot[i, j] = np.ma.masked
            else:
                # If the data is masked, then mask the grid point in the output.
                self.rhodot[i, j] = np.ma.masked
        return self.rhodot


class TrajectoryRepulsionRatio(EulerianDiagnostic2D):
    """ Computes and stores the nudot field for a 2 dimensional flow.
    """
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...] = None,
        v: np.ndarray[np.ndarray[float, ...], ...] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """ Computes the rhodot field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
            Returns:
                rhodot (np.ndarray): The rhodot field.
                nudot (np.ndarray): The nudot field.
        """
        super().compute(x, y, u, v, f, t)

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.nudot = np.ma.empty([self.ydim, self.xdim])
        J = np.array([[0, 1], [-1, 0]])
        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                Velocity = np.array([self.u[i, j], self.v[i, j]])
                Velocity_Squared = np.dot(Velocity, Velocity)
                if Velocity_Squared:
                    self.nudot[i, j] = np.dot(
                        Velocity, np.dot(np.trace(S) * np.identity(2) - 2 * S, Velocity)
                    ) / Velocity_Squared
                else:
                    # If V dot V = 0, then mask the grid point in the output.
                    self.nudot[i, j] = np.ma.masked
            else:
                # If the data is masked, then mask the grid point in the output.
                self.nudot[i, j] = np.ma.masked
        return self.nudot


class iLES(EulerianDiagnostic2D, RidgeExtractor2D):
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...] = None,
        v: np.ndarray[np.ndarray[float, ...], ...] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: tuple[float, float] = None,
        kind: str = 'attacting',
        edge_order: int = 1,
        percentile: float = None,
        force_eigenvectors: bool = False,
        debug: bool = False
    ) -> np.ndarray[np.ndarray[float, ...], ...]:
        if kind.lower() == 'attracting':
            eig_i = 0
        elif kind.lower() == 'repelling':
            eig_i = -1
        else:
            raise ValueError(
                f'kind: {kind}, unrecognized, please use either "attracting" or "repelling"'
            )
        super().compute(x, y, u, v, f, t)

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.rate_field = np.ma.empty([self.ydim, self.xdim])
        self.Xi_max = np.ma.empty([self.ydim, self.xdim, 2])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                eigenValues, eigenVectors = np.linalg.eig(S)
                idx = eigenValues.argsort()
                self.rate_field[i, j] = eigenValues[idx[eig_i]]
                self.Xi_max[i, j, :] = eigenVectors[:, idx[eig_i]]

            else:
                # If the data is masked, then mask the grid point in the output.
                self.rate_field[i, j] = np.ma.masked
                self.Xi_max[i, j, 0] = np.ma.masked
                self.Xi_max[i, j, 1] = np.ma.masked

        # derivatives are no longer needed deleting to be more space efficent and allow larger
        # fields.
        del dudx, dudy, dvdx, dvdy

        if kind == 'attracting':
            self.rate_field = -self.rate_field

        if force_eigenvectors:
            self.Xi_max = force_eigenvectors2D(self.Xi_max)

        self.iles = self.extract_ridges(self.rate_field, self.Xi_max, percentile, edge_order, debug)

        # free up some extra memory if not needed
        if not debug:
            self.Xi_max
        return self.iles
