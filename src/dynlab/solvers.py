from abc import ABC, abstractmethod
import numpy as np
import scipy.integrate as sint


class Solver(ABC):
    @abstractmethod
    def compute(self):
        pass

class FlowMapSolver(Solver):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, f, t, x, y, **kwargs):
        if len(t) != 2:
            raise ValueError("t must only have 2 values, t_0 and t_final")
        
        self.y = (np.array(y) if type(y) is not np.ndarray else y).squeeze()
        self.x = (np.array(x) if type(x) is not np.ndarray else x).squeeze()

        if len(self.y.shape) > 1:
            raise ValueError("y must be a 1-dimensional array.")
 
        if len(self.x.shape) > 1:
            raise ValueError("x must be a 1-dimensional array.")

        self.ydim = len(self.y)
        self.xdim = len(self.x)
        flow_map = np.empty([self.ydim*self.xdim, 2])
    
        #integrate velocity field
        idx = 0
        for y0 in self.y:
            for x0 in self.x:
                sol = sint.solve_ivp(
                    f, t, (x0, y0), **kwargs
                )
                flow_map[idx,:] = sol.y[:,-1]
                idx += 1
        
        self.flow_map = np.reshape(flow_map, [self.ydim, self.xdim, -1])
        return self.flow_map
    
class FTLESolver(FlowMapSolver):
    def __init__(self) -> None:
        super().__init__()

    def compute(self, f, t, x, y, edge_order=1, **kwargs):
        super().compute(f, t, x, y, **kwargs) 
        fx,fy = np.split(self.flow_map, 2, -1)

        #Calculate flow map gradients
        dfxdy,dfxdx = np.gradient(fx.squeeze(), self.y, self.x, edge_order=edge_order)
        dfydy,dfydx = np.gradient(fy.squeeze(), self.y, self.x, edge_order=edge_order)
        del fx, fy

        sigma = np.empty([self.ydim, self.xdim])
        for i in range(self.ydim):
            for j in range(self.xdim):
                #Calculate Cauchy-Green tensor, C
                JF = np.array([[dfxdx[i,j],dfxdy[i,j]],[dfydx[i,j],dfydy[i,j]]])
                C = np.dot(JF.T, JF)
                
                #Calculate FTLE, sigma
                lam=np.max(np.linalg.eig(C)[0])
                if lam>=1:
                    sigma[i,j]=1.0/(2.0*abs(t[-1]-t[0]))*np.log(lam)
                else:
                    sigma[i,j]=0
                    
        return sigma
  

x = [[0,1,2]]
y = [0,1]
# x, y = np.meshgrid(x,y)
from dynlab.velocity_fields import double_gyre
from dynlab.solvers import FTLESolver
print(FTLESolver().compute(double_gyre, (2, 0), x, y))


