import warnings
import numpy as np
from typing import Callable
from itertools import product
from abc import ABC, abstractmethod

from contourpy import contour_generator
from pathos.multiprocessing import ProcessingPool as Pool

from dynlab.utils import odeint_wrapper


class Diagnostic2D(ABC):
    """ Diagnostic base class for 2D flows. """
    def __init__(self, num_threads: int = 1) -> None:
        """ Initializes diagnostic base class, including the setting up of multiprocessing when
                requested.
            Args:
                num_threads (int): The number of threads to use for multiprocessing. Defaults to 1.
            Returns:
                None
        """
        super().__init__()
        self.num_threads = num_threads
        if num_threads == 1:
            self.map = map
        elif num_threads > 1:
            def pool_wrapper(f, data):
                with Pool(num_threads) as p:
                    result = p.map(f, data)
                return result
            self.map = pool_wrapper
        else:
            raise ValueError("num_threads must be an integer greater than 0.")

    @abstractmethod
    def compute(self, x: np.ndarray[float], y: np.ndarray[float]) -> None:
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

    @staticmethod
    def not_masked(*args):
        """ staticmethod to ensure the derivatives are not masked before calculating the strain or
            rate-of-strain tensors.
            Args:
                *args: values to check for masks.
            Returns:
                (bool): True when none of the args are masked, false when at least 1 value is
                    masked.
        """
        # this function works because sums with masked values return masked results. So if args[3]
        # is masked then the final result of the comparison will be masked.
        return sum(args) is not np.ma.masked


class EulerianDiagnostic2D(Diagnostic2D, ABC):
    """ Eulerian diagnostic base class for 2D flows. """
    @abstractmethod
    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        u: np.ndarray[np.ndarray[float]],
        v: np.ndarray[np.ndarray[float]],
        f: Callable[[float, tuple[float, float]], tuple] = None,
        t: float = None,
        edge_order: int = 1
    ) -> tuple[
        np.ndarray[np.ndarray[float]],
        np.ndarray[np.ndarray[float]],
        np.ndarray[np.ndarray[float]],
        np.ndarray[np.ndarray[float]]
    ]:
        """ base 2D eulerian compute function. Ensures that u and v are np.array that they have the
                proper length. Also sets class attributes: u, v; and calculates the gradients of
                the velocity fields.
            Args:
                x (np.ndarray): the x coordinates of the flow.
                y (np.ndarray): the y coordinates of the flow.
                u (np.ndarray): the u velocity component of the flow.
                v (np.ndarray): the v velocity component of the flow.
                f (function): the vector function from which to calculate u and v, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (float): the time step at which f will calculate u and v.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
            Returns:
                dudy (np.ndarray): The partial derivative of u with respect to y
                dudx (np.ndarray): The partial derivative of u with respect to x
                dvdy (np.ndarray): The partial derivative of v with respect to y
                dvdx (np.ndarray): The partial derivative of v with respect to x
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

        # Calculate the gradients of the velocity field
        dudy, dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy, dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        return dudy, dudx, dvdy, dvdx


class LagrangianDiagnostic2D(Diagnostic2D, ABC):
    """ Lagrangian diagnostic base class for 2D flows. """
    def __init__(
        self,
        integrator: Callable = odeint_wrapper,
        num_threads: int = 1
    ) -> None:
        """ Initializes lagrangian diagnostic base class.
            Args:
                integrator (Callable): function for integrating velocity fields. Must conform to
                    the syntax f(t, Y), where t is the time-step and Y is the position vector.
                    Defaults to a wrapped verion of scipy's odeint, which itself is a wrapper for
                    LSODA.
            Returns:
                None
        """
        super().__init__(num_threads)
        self.integrator = integrator

    @abstractmethod
    def compute(
        self,
        x: np.ndarray[float],
        y: np.ndarray[float],
        f: Callable[[float, tuple[float, float]], tuple],
        t: tuple[float, float],
        **kwargs
    ) -> np.ndarray[np.ndarray[float]]:
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
                **kwargs: keyword arguments for the integrator.
            Returns:
                flow_map (np.ndarray): The final position of the trajectories.
        """
        super().compute(x, y)

        if len(t) != 2:
            raise ValueError("t must only have 2 values, t_0 and t_final")
        # calculate the flow map
        flow_map = np.array(list(
            self.map(
                lambda initial_values: self.integrator(
                    f, t, initial_values[::-1], **kwargs
                )[-1, :],
                product(self.y, self.x)
            )
        ))

        flow_map = flow_map.reshape([self.ydim, self.xdim, 2])

        return flow_map


class RidgeExtractor2D(Diagnostic2D, ABC):
    """ Encapsulates 2D ridge extraction code. """
    def extract_ridges(
        self,
        field: np.ndarray[np.ndarray[float]],
        Xi: np.ndarray[np.ndarray[np.ndarray[float]]],
        percentile: int,
        edge_order: int,
        debug: bool = False
    ):
        """ Calculates and extracts ridges perpendicular to the directions of maximum stretching
                in 2D flows.
            Args:
                field (np.ndarray): the field in which to find ridges.
                Xi (np.ndarray): the eigenvector field of the given field.
                percentile (float): which percentile to filter LCS on, i.e. 90 would filter out
                    LCS weaker than the 90th percentile. Defaults to None.
                edge_order (int): order to use for gradient calculation. Defaults to 1.
                debug (bool): when True algorithm will store the directional derivative field
                    (directional_derivative) and the concavity field (concavity) to allow users
                    to dig deeper into the LCS results.
            Returns:
                ridge_lines (np.ndarray): collection of ridge coordinates.
        """
        # Calculate gradients of the ftle field
        dfdy, dfdx = np.gradient(field, self.y, self.x, edge_order=edge_order)
        dfdydy, dfdydx = np.gradient(dfdy, self.y, self.x, edge_order=edge_order)
        dfdxdy, dfdxdx = np.gradient(dfdx, self.y, self.x, edge_order=edge_order)

        # initialize directional derivative and concavity matrices
        self.directional_derivative = np.ma.empty([self.ydim, self.xdim])
        self.concavity = np.ma.empty([self.ydim, self.xdim])

        for i, j in product(range(self.ydim), range(self.xdim)):
            # Make sure the data is not masked, masked gridpoints do not work with
            # Python's linalg module

            if self.not_masked(
                dfdx[i, j], dfdy[i, j], dfdxdy[i, j],
                dfdydy[i, j], dfdxdx[i, j], dfdydx[i, j]
            ):
                # compute the directional derivative and the concavity
                self.directional_derivative[i, j] = np.dot(
                    [dfdx[i, j], dfdy[i, j]], Xi[i, j, :]
                )
                self.concavity[i, j] = np.dot(
                    np.dot(
                        [
                            [dfdxdx[i, j], dfdxdy[i, j]],
                            [dfdydx[i, j], dfdydy[i, j]]
                        ], Xi[i, j, :]
                    ), Xi[i, j, :])
            else:
                self.directional_derivative[i, j] = np.ma.masked
                self.concavity[i, j] = np.ma.masked

        # free up some extra memory
        del dfdy, dfdx, dfdydy, dfdydx, dfdxdy, dfdxdx

        self.directional_derivative = np.ma.masked_where(
            field <= 0, self.directional_derivative
        )

        # mask the directional derivative field where ever the f is concave up.
        self.directional_derivative = np.ma.masked_where(
            self.concavity >= 0, self.directional_derivative
        )

        if percentile:
            # there's a warning here about it using the masked field, which is useless because
            # that's exactly what we want it to do, so we suppress this warning so as to not
            # confuse the user. Warnings after this block will still apear.
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                self.directional_derivative = np.ma.masked_where(
                    field <= np.percentile(field, percentile), self.directional_derivative
                )

        ridge_lines = contour_generator(x=self.x, y=self.y, z=self.directional_derivative).lines(0)

        # free up some extra memory if not needed
        if not debug:
            del self.directional_derivative
            del self.concavity

        return ridge_lines
