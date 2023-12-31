from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import scipy.integrate as sint


class Diagnostic2D(ABC):
    """ Diagnostic base class for 2D flows."""
    @abstractmethod
    def compute(self, x: np.ndarray[float, ...], y: np.ndarray[float, ...]) -> None:
        """ base 2D compute function. Ensures that x and y are np.array that they have the proper
                length. Also sets class attributes: x, y, xdim, ydim.
            Args:
                x (float): the x coordinates of the flow.
                y (float): the y coordinates of the flow.
            Returns:
                None
        """
        # cast lists / tuples to np.ndarray to add flexibility.
        self.y = (np.array(y) if type(y) is not np.ndarray else y).squeeze()
        self.x = (np.array(x) if type(x) is not np.ndarray else x).squeeze()

        if len(self.y.shape) > 1:
            raise ValueError("y must be a 1-dimensional array.")

        if len(self.x.shape) > 1:
            raise ValueError("x must be a 1-dimensional array.")

        self.ydim = len(self.y)
        self.xdim = len(self.x)


class EulerianDiagnostic2D(Diagnostic2D, ABC):
    """ Eulerian diagnostic base class for 2D flows."""
    @abstractmethod
    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        u: np.ndarray[np.ndarray[float, ...], ...],
        v: np.ndarray[np.ndarray[float, ...], ...],
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None
    ) -> None:
        """ base 2D eulerian compute function. Ensures that u and v are np.array that they have the
                proper length. Also sets class attributes: u, v. NOTE: checks for x and y are
                handled by the parent class compute method.
            Args:
                x (np.ndarray): the x coordinates of the flow.
                y (np.ndarray): the y coordinates of the flow.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
            Returns:
                None
        """
        super().compute(x, y)
        if u is not None and v is not None:
            # cast lists / tuples to np.ndarray to add flexibility.
            self.v = (np.array(v) if type(v) is not np.ndarray else v).squeeze()
            self.u = (np.array(u) if type(u) is not np.ndarray else u).squeeze()

            if len(self.v.shape) != 2:
                raise ValueError("v must be a 2-dimensional array.")

            if len(self.u.shape) != 2:
                raise ValueError("u must be a 2-dimensional array.")
        elif f is not None and t is not None:
            self.u, self.v = f(t, np.meshgrid(x, y))
        else:
            raise RuntimeError(
                "Either a velocity field (u, v) or a function and timestep from which to "
                + "calculate a velocity field must be passed to the compute function."
            )


class LagrangianDiagnostic(Diagnostic2D):
    """ Calculates and stores the flow map for a 2 dimensional flow."""
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self,
        x: np.ndarray[float, ...],
        y: np.ndarray[float, ...],
        f: Callable[[float, tuple[float, float]], tuple],
        t: tuple[float, float],
        **kwargs
    ) -> np.ndarray[np.ndarray[float, ...], ...]:
        """ Computes the flow map for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the initial x-coordinates for the
                    trajectories.
                y (np.ndarray): 1-d array containing the initial y-coordinates for the
                    trajectories.
                f (function): the vector function to calculate trajectories in, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (tuple): the time interval over which to calculate trajectories, t0 to tf.
                NOTE: function also accepts kwargs for scipy.integrate.solve_ivp.
            Returns:
                flow_map (np.ndarray): The final position of the trajectories.
        """
        super().compute(x, y)

        if len(t) != 2:
            raise ValueError("t must only have 2 values, t_0 and t_final")

        # initialize flow map
        self.flow_map = np.empty([self.ydim, self.xdim, 2])

        # integrate velocity field
        idx = 0
        for i, y0 in enumerate(self.y):
            for j, x0 in enumerate(self.x):
                sol = sint.solve_ivp(
                    f, t, (x0, y0), **kwargs
                )
                self.flow_map[i, j,:] = sol.y[:,-1]
                idx += 1

        return self.flow_map


class FTLE(LagrangianDiagnostic):
    """ Calculates and stores the FTLE field for a 2 dimensional flow."""
    def __init__(self) -> None:
        super().__init__()

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

        for i in range(self.ydim):
            for j in range(self.xdim):
                # Make sure the data is not masked, masked gridpoints do not work with
                # Python's linalg module
                if (
                    dfxdx[i, j] and dfxdy[i, j] and dfydx[i, j] and dfydy[i, j]
                ) is not np.ma.masked:
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


class AttractionRate(EulerianDiagnostic2D):
    """ Computes and stores the attraction and repulsion rate fields (s_1 & s_n) for a
            2 dimensional flow.
    """
    def __init__(self) -> None:
        super().__init__()

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

        for i in range(self.ydim):
            for j in range(self.xdim):
                # Make sure the data is not masked, masked gridpoints do not work with
                # Python's linalg module
                if (dudx[i, j] and dudy[i, j] and dvdx[i, j] and dvdy[i, j]) is not np.ma.masked:
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


class Rhodot(EulerianDiagnostic2D):
    """ Computes and stores the rhodot field for a 2 dimensional flow.
    """
    def __init__(self) -> None:
        super().__init__()

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
        self.rhodot = np.ma.empty([self.ydim, self.xdim])
        self.nudot = np.ma.empty([self.ydim, self.xdim])
        J = np.array([[0, 1], [-1, 0]])
        for i in range(self.ydim):
            for j in range(self.xdim):
                # Make sure the data is not masked, masked gridpoints do not work with
                # Python's linalg module
                if (dudx[i, j] and dudy[i, j] and dvdx[i, j] and dvdy[i, j]) is not np.ma.masked:
                    # If the data is not masked, compute s_1 and s_n
                    Gradient = np.array([[dudx[i, j], dudy[i, j]], [dvdx[i, j], dvdy[i, j]]])
                    S = 0.5*(Gradient + np.transpose(Gradient))
                    Velocity = np.array([self.u[i, j], self.v[i, j]])
                    Velocity_Squared = np.dot(Velocity, Velocity)
                    if Velocity_Squared:
                        self.rhodot[i, j] = np.dot(
                            Velocity, np.dot(np.matmul(J.T, np.matmul(S, J)), Velocity)
                        ) / Velocity_Squared
                        self.nudot[i, j] = np.dot(
                            Velocity, np.dot(np.trace(S) * np.identity(2) - 2 * S, Velocity)
                        ) / Velocity_Squared
                    else:
                        # If V dot V = 0, then mask the grid point in the output.
                        self.rhodot[i, j] = np.ma.masked
                        self.nudot[i, j] = np.ma.masked
                else:
                    # If the data is masked, then mask the grid point in the output.
                    self.rhodot[i, j] = np.ma.masked
                    self.nudot[i, j] = np.ma.masked
        return self.rhodot, self.nudot
