import numpy as np
from typing import Callable
from itertools import product

from dynlab.diagnostics._base_classes import EulerianDiagnostic2D, RidgeExtractor2D
from dynlab.utils import force_eigenvectors2D


class _AttractionRepulsionRate(EulerianDiagnostic2D):
    """ Computes and stores the attraction and repulsion rate fields (s_1 & s_n) for a
            2 dimensional flow.
    """
    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1,
        eigenvalue_index: int = 0
    ) -> np.ndarray[np.ndarray[float]]:
        """ Computes the attraction rate or repulsion rate fields for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
                eigenvalue_index (int): Which eigenvalue (from smallest to largest) to return.
                    Defaults to 0 (smallest).
            Returns:
                field (np.ndarray): The field for the selected eigenvector.
        """
        dudy, dudx, dvdy, dvdx = super().compute(x, y, u, v, f, t, edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        field = np.ma.empty([self.ydim, self.xdim])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module
            if self.not_masked(dudx[i, j], dudy[i, j], dvdx[i, j], dvdy[i, j]):
                # If the data is not masked, compute s_1 and s_n
                Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                S = 0.5*(Gradient + np.transpose(Gradient))
                eigenValues, _ = np.linalg.eig(S)
                idx = eigenValues.argsort()
                field[i, j] = eigenValues[idx[eigenvalue_index]]
            else:
                # If the data is masked, then mask the grid point in the output.
                field[i, j] = np.ma.masked

        return field


class AttractionRate(_AttractionRepulsionRate):
    """ Computes and stores the attraction rate field for a 2 dimensional flow. """
    def __init__(self, num_threads: int = 1) -> None:
        """ Initializes class object.
            num_threads (int): The number of threads to process on. Defaults to 1 (single threaded).
        """
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> np.ndarray[np.ndarray[float]]:
        """ Computes the attraction rate field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
            Returns:
                field (np.ndarray): The attraction rate field.
        """
        self.field = super().compute(x, y, u, v, f, t, edge_order, 0)
        return self.field


class RepulsionRate(_AttractionRepulsionRate):
    """ Computes and stores the repulsion rate field for a 2 dimensional flow. """
    def __init__(self, num_threads: int = 1) -> None:
        """ Initializes class object.
            num_threads (int): The number of threads to process on. Defaults to 1 (single threaded).
        """
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> np.ndarray[np.ndarray[float]]:
        """ Computes the repulsion rate field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
            Returns:
                field (np.ndarray): The repulsion rate field.
        """
        self.field = super().compute(x, y, u, v, f, t, edge_order, -1)
        return self.field


class _TrajectoryRepulsionRateRatio(EulerianDiagnostic2D):
    """ Computes the trajectory repulsion rate field and the trajectory repulsion ratio field for
        a 2 dimensional flow.
    """
    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1,
        rate_or_ratio: str = 'rate'
    ) -> np.ndarray[np.ndarray[float]]:
        """ Computes the trajectory repulsion rate field or the trajectory repulsion ratio field
                for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
                rate_or_ratio (str): indicates whether the function could compute the repulsion
                    rate or the repulsion ratio. Defaults to "rate".
            Returns:
                field (np.ndarray): The trajectory repulsion rate field or the trajectory repulsion
                    ratio field.
        """
        dudy, dudx, dvdy, dvdx = super().compute(x, y, u, v, f, t, edge_order)

        if rate_or_ratio == 'rate':
            def rate_ratio_func(Velocity, S, Velocity_Squared):
                J = np.array([[0, 1], [-1, 0]])
                return (
                    np.dot(Velocity, np.dot(np.matmul(J.T, np.matmul(S, J)), Velocity))
                    / Velocity_Squared
                )
        elif rate_or_ratio == 'ratio':
            def rate_ratio_func(Velocity, S, Velocity_Squared):
                return (
                    np.dot(Velocity, np.dot(np.trace(S) * np.identity(2) - 2 * S, Velocity))
                    / Velocity_Squared
                )
        else:
            raise ValueError('rate_or_ratio parameter must be either "rate" or "ratio".')

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        field = np.ma.empty([self.ydim, self.xdim])
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
                    field[i, j] = rate_ratio_func(Velocity, S, Velocity_Squared)
                else:
                    # If V dot V = 0, then mask the grid point in the output.
                    field[i, j] = np.ma.masked
            else:
                # If the data is masked, then mask the grid point in the output.
                field[i, j] = np.ma.masked
        return field


class TrajectoryRepulsionRate(_TrajectoryRepulsionRateRatio):
    """ Computes and stores the trajectory repulsion rate field for a 2 dimensional flow. """
    def __init__(self, num_threads: int = 1) -> None:
        """ Initializes class object.
            num_threads (int): The number of threads to process on. Defaults to 1 (single threaded).
        """
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> np.ndarray[np.ndarray[float]]:
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
                edge_order (int): order to use for gradient calculation. Defaults to 1.
            Returns:
                field (np.ndarray): The trajectory repulsion rate field.
        """
        self.field = super().compute(x, y, u, v, f, t, edge_order, 'rate')
        return self.field


class TrajectoryRepulsionRatio(_TrajectoryRepulsionRateRatio):
    """ Computes and stores the trajectory repulsion ratio field for a 2 dimensional flow. """
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> np.ndarray[np.ndarray[float]]:
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
                edge_order (int): order to use for gradient calculation. Defaults to 1.
            Returns:
                field (np.ndarray): The trajectory repulsion ratio field.
        """
        self.field = super().compute(x, y, u, v, f, t, edge_order, 'ratio')
        return self.field


class iLES(EulerianDiagnostic2D, RidgeExtractor2D):
    """ Computes and stores the iLES for a 2 dimensional flow. """
    def __init__(self, num_threads: int = 1) -> None:
        super().__init__(num_threads)

    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]] = None,
        v: np.ndarray[np.ndarray[float]] = None,
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: tuple[float, float] = None,
        kind: str = 'attacting',
        edge_order: int = 1,
        percentile: float = None,
        force_eigenvectors: bool = False,
        debug: bool = False
    ) -> np.ndarray[np.ndarray[float]]:
        """ Computes the LCS for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate the FTLE, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                kind (str): indicates whether the iles should be attracting or repelling. Defaults
                    to attracting.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
                percentile (float): which percentile to filter iLES on, i.e. 90 would filter out
                    iLES weaker than the 90th percentile. Defaults to None.
                force_eigenvectors (bool): np.linalg.eig will product eigenvectors unique up to
                    their sign. This flag forces eigenvectors to have the same sign. Defaults to
                    False.
                debug (bool): when True algorithm will store the eigenvector field (Xi_max), the
                    directional derivative field (directional_derivative) and the concavity field
                    (concavity) to allow users to dig deeper into the LCS results.
                **kwargs: keyword arguments for the integrator.
            Returns:
                lcs (np.ndarray): collection of iles coordinates.
        """
        if kind.lower() == 'attracting':
            eig_i = 0
        elif kind.lower() == 'repelling':
            eig_i = -1
        else:
            raise ValueError(
                f'kind: {kind}, unrecognized, please use either "attracting" or "repelling"'
            )

        dudy, dudx, dvdy, dvdx = super().compute(x, y, u, v, f, t, edge_order)

        # Initialize arrays for the attraction rate and repullsion rate
        # Using masked arrays can be very useful when dealing with geophysical data and
        # data with gaps in it.
        self.field = np.ma.empty([self.ydim, self.xdim])
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
                self.field[i, j] = eigenValues[idx[eig_i]]
                self.Xi_max[i, j, :] = eigenVectors[:, idx[eig_i]]

            else:
                # If the data is masked, then mask the grid point in the output.
                self.field[i, j] = np.ma.masked
                self.Xi_max[i, j, 0] = np.ma.masked
                self.Xi_max[i, j, 1] = np.ma.masked

        # derivatives are no longer needed deleting to be more space efficent and allow larger
        # fields.
        del dudx, dudy, dvdx, dvdy

        if kind == 'attracting':
            self.field = -self.field

        if force_eigenvectors:
            self.Xi_max = force_eigenvectors2D(self.Xi_max)

        self.iles = self.extract_ridges(self.field, self.Xi_max, percentile, edge_order, debug)

        # free up some extra memory if not needed
        if not debug:
            del self.Xi_max

        return self.iles
