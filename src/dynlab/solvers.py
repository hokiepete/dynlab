from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import scipy.integrate as sint


class Solver2D(ABC):
    @abstractmethod
    def compute(self, x, y):
        
        # cast lists / tuples to np.ndarray to add flexibility.
        self.y = (np.array(y) if type(y) is not np.ndarray else y).squeeze()
        self.x = (np.array(x) if type(x) is not np.ndarray else x).squeeze()

        if len(self.y.shape) > 1:
            raise ValueError("y must be a 1-dimensional array.")
 
        if len(self.x.shape) > 1:
            raise ValueError("x must be a 1-dimensional array.")

        self.ydim = len(self.y)
        self.xdim = len(self.x)


class EulerianSolver2D(Solver2D, ABC):
    @abstractmethod
    def compute(self, u, v, x, y):
        super().compute(x, y)
        
        # cast lists / tuples to np.ndarray to add flexibility.
        self.v = (np.array(v) if type(v) is not np.ndarray else v).squeeze()
        self.u = (np.array(u) if type(u) is not np.ndarray else u).squeeze()

        if len(self.v.shape) != 2:
            raise ValueError("v must be a 2-dimensional array.")
 
        if len(self.x.shape) != 2:
            raise ValueError("u must be a 2-dimensional array.")
    

class FlowMapSolver(Solver2D):
    """ Calculates and stores the flow map for a 2 dimensional flow. """
    def __init__(self) -> None:
        super().__init__()

    def compute(
            self,
            f: Callable[[float, tuple[float, float]], list],
            t: tuple[float, float],
            x: np.ndarray[float, ...], 
            y: np.ndarray[float, ...], 
            **kwargs
        ) -> np.ndarray:
        """ Computes the flow map for a given vector field.
            Args:
                f (function): the vector function to calculate trajectories in, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (tuple): the time interval over which to calculate trajectories, t0 to tf.
                x (np.ndarray): 1-d array containing the initial x-coordinates for the
                    trajectories. 
                y (np.ndarray): 1-d array containing the initial y-coordinates for the
                    trajectories. 
                NOTE: function also accepts kwargs for scipy.integrate.solve_ivp.
            Returns:
                flow_map (np.ndarray): The final position of the trajectories.
        """
        super().compute(x,y)

        if len(t) != 2:
            raise ValueError("t must only have 2 values, t_0 and t_final")
        
        # initialize flow map
        self.flow_map = np.empty([self.ydim, self.xdim, 2])
    
        #integrate velocity field
        idx = 0
        for i, y0 in enumerate(self.y):
            for j, x0 in enumerate(self.x):
                sol = sint.solve_ivp(
                    f, t, (x0, y0), **kwargs
                )
                self.flow_map[i, j,:] = sol.y[:,-1]
                idx += 1
        
        return self.flow_map


class FTLESolver(FlowMapSolver):
    """ Calculates and stores the FTLE field for a 2 dimensional flow. """
    def __init__(self) -> None:
        super().__init__()

    def compute(self,
            f: Callable[[float, tuple[float, float]], list],
            t: tuple[float, float],
            x: np.ndarray[float, ...],
            y: np.ndarray[float, ...],
            edge_order: int=1,
            **kwargs
        ) -> np.ndarray:
        """ Computes the FTLE field for a given vector field.
            Args:
                f (function): the vector function from which to calculate the FTLE, f must take 2
                    arguments time (scalar) and position (vector), e.g. f(t, Y) where Y contains
                    the x position and the y position [x, y].
                t (tuple): the time interval over which to calculate FTLE values, t0 to tf.
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field. 
                NOTE: function also accepts kwargs for scipy.integrate.solve_ivp.
            Returns:
                flow_map (np.ndarray): The final position of the trajectories.
        """
        
        # computes the flow map
        super().compute(f, t, x, y, **kwargs) 
        
        #Calculate flow map gradients
        dfxdy,dfxdx = np.gradient(
            self.flow_map[:, :, 0].squeeze(), self.y, self.x, edge_order=edge_order
        )
        dfydy,dfydx = np.gradient(
            self.flow_map[:, :, 1].squeeze(), self.y, self.x, edge_order=edge_order
        )
        
        # initialize FTLE matrix
        self.sigma = np.ma.empty([self.ydim, self.xdim])
        
        for i in range(self.ydim):
            for j in range(self.xdim):
                #Make sure the data is not masked, masked gridpoints do not work with
                #Python's linalg module
                if (dfxdx[i,j] and dfxdy[i,j] and dfydx[i,j] and dfydy[i,j]) is not np.ma.masked:
                    #Calculate Cauchy-Green tensor, C
                    JF = np.array([[dfxdx[i,j],dfxdy[i,j]],[dfydx[i,j],dfydy[i,j]]])
                    C = np.dot(JF.T, JF)
                    
                    #Calculate FTLE, sigma
                    lambda_max=np.max(np.linalg.eig(C)[0])
                    if lambda_max>=1:
                        self.sigma[i,j] = 1.0/(2.0*abs(t[-1]-t[0]))*np.log(lambda_max)
                    else:
                        self.sigma[i,j] = 0
                else:
                    #If the data is masked, then mask the grid point in the output.
                    self.sigma[i,j] = np.ma.masked
                        
        return self.sigma


class AttractionRateSolver(EulerianSolver2D):
    """ Computes and stores the attraction and repulsion rate fields (s_1 & s_n) for a 2 
        dimensional flow.
    """
    def __init__(self) -> None:
        super().__init__()

    def compute(self,
            u: np.ndarray[np.ndarray[float, ...], ...],
            v: np.ndarray[np.ndarray[float, ...], ...],
            x: np.ndarray[float, ...],
            y: np.ndarray[float, ...],
            edge_order: int=1
        ) -> tuple[np.ndarray, np.ndarray]:
        """ Computes the FTLE field for a given vector field.
            Args:
                x (np.ndarray): 1-d array containing the x-coordinates of the field.
                y (np.ndarray): 1-d array containing the y-coordinates of the field. 
                NOTE: function also accepts kwargs for scipy.integrate.solve_ivp.
            Returns:
                flow_map (np.ndarray): The final position of the trajectories.
        """
        super.compute(u, v, x, y)

        #Calculate the gradients of the velocity field
        dudy,dudx = np.gradient(self.u, self.y, self.x, edge_order=edge_order)
        dvdy,dvdx = np.gradient(self.v, self.y, self.x, edge_order=edge_order)

        #Initialize arrays for the attraction rate and repullsion rate
        #Using masked arrays can be very useful when dealing with geophysical data and
        #data with gaps in it.
        self.s1 = np.ma.empty([self.ydim, self.xdim])
        self.sn = np.ma.empty([self.ydim, self.xdim])

        for i in range(self.ydim):
            for j in range(self.xdim):
                #Make sure the data is not masked, masked gridpoints do not work with
                #Python's linalg module
                if (dudx[i,j] and dudy[i,j] and dvdx[i,j] and dvdy[i,j]) is not np.ma.masked:
                    #If the data is not masked, compute s_1 and s_n
                    Grad = np.array([[dudx[i,j], dudy[i,j]], [dvdx[i,j], dvdy[i,j]]])
                    S = 0.5*(Grad + np.transpose(Grad))
                    eigenValues, _ = np.linalg.eig(S)
                    idx = eigenValues.argsort()
                    self.s1[i,j] = eigenValues[idx[0]]
                    self.sn[i,j] = eigenValues[idx[-1]]

                else:
                    #If the data is masked, then mask the grid point in the output.
                    self.s1[i,j] = np.ma.masked
                    self.sn[i,j] = np.ma.masked




x = [[0,1,2]]
y = [0,1]
# x, y = np.meshgrid(x,y)
from dynlab.velocity_fields import double_gyre
from dynlab.solvers import FTLESolver
print(FTLESolver().compute(double_gyre, (2, 0), x, y))


