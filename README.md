# dynlab

Dynlab is a python package to make dynamical systems analysis faster and easier, by providing practitioners prebuilt modules for calculating common diagnostics. Dynlab currently offers several easy to use diagnostics for analyzing your dynamical systems including an FTLE calculator `FLTE`, an attraction and repulsion rate calculator `AttractionRate`, and a trajectory repulsion rate calculator `TrajectoryRepulsionRate`. Dynlab also offers users a variety of prewritten flows to experiment and play around with, including the double gyre `double_gyre`, the Duffing oscillator `duffing_oscillator`, the pendulum with damping and forcing `pendulum`, the Van Der Pol oscillator `van_der_pol_oscillator`, the Lotka-Voltera flow `lotka_volterra`, the Lorenz system `lorenz`, and many more. Note that all the flows are written in such a way that they can easily be passed to an ode integrator and as such even the autonomous flows will still require a time parameter to be passed to them.

Currently supported Lagrangian diagnostics are:\
    Finite-Time Lyapunov Exponent: `FTLE`\
    Lagrangian Coherent Structures: `LCS`

Currently supported Eulerian diagnostics are:\
    Attraction Rate: `AttractionRate`\
    Repulsion Rate: `RepulsionRate`\
    infinitesimal Lyapunov ExponentS: `iLES`\
    Trajectory Repulsion Rate: `TrajectoryRepulsionRate`\
    Trajectory Repulsion Ratio: `TrajectoryRepulsionRatio`

This package is readily available on pypi and can be easily installed with the command `pip install dynlab`.
Note that dynlab is written with python 3.11 and you may need to update your python installation to take advantage of it.

# Example
```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import FTLE
from dynlab.flows import double_gyre
x = np.linspace(0, 2, 101)
y = np.linspace(0, 1, 101)
ftle = FTLE(num_threads=1).compute(x, y, double_gyre, (10, 0), edge_order=2, rtol=1e-8, atol=1e-8)
plt.pcolormesh(x, y, ftle, shading='gouraud')
```
![alt text](https://github.com/hokiepete/docs/blob/main/images/double_gyre_ftle.png)

```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import LCS
from dynlab.flows import double_gyre
x = np.linspace(0, 2, 201)
y = np.linspace(0, 1, 101)
lcs = LCS()
attracting_lcs = lcs.compute(x, y, f=double_gyre, t=(10, 0), percentile=80)
for line in attracting_lcs:
    plt.plot(line[:, 0],line[:, 1], 'b')
```
![alt text](https://github.com/hokiepete/docs/blob/main/images/double_gyre_lcs.png)

```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import iLES
from dynlab.flows import double_gyre
x = np.linspace(0, 2, 201)
y = np.linspace(0, 1, 101)
attracting_iles = iLES().compute(x, y, f=double_gyre, t=0, kind='attracting', force_eigenvectors=True)
repelling_iles = iLES().compute(x, y, f=double_gyre, t=0, kind='repelling', force_eigenvectors=True)

xx, yy = np.meshgrid(x[::10],y[::10])
u, v = double_gyre(0, (xx, yy))
plt.quiver(xx,yy,u,v)
for line in attracting_iles:
    plt.plot(line[:, 0],line[:, 1], 'b', linewidth=3)
for line in repelling_iles:
    plt.plot(line[:, 0],line[:, 1], 'r', linewidth=3)
```
![alt text](https://github.com/hokiepete/docs/blob/main/images/double_gyre_iles.png)


```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import AttractionRate
from dynlab.flows import double_gyre
x = np.linspace(0, 20000, 101)
y = np.linspace(-4000, 4000, 101)
u, v = bickley_jet(0, np.meshgrid(x, y))
attraction_rate = AttractionRate().compute(x, y, u=u, v=v, edge_order=2)
# note that lower values of the attraction rate field equate to higher levels of attraction
# so we'll plot the negative of the attraction rate field to highlight areas of greatest attraction.
plt.pcolormesh(x, y, -attraction_rate, shading='gouraud')
```
![alt text](https://github.com/hokiepete/docs/blob/main/images/bickley_jet_attraction_rate.png)

```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import Rhodot
from dynlab.flows import bead_on_a_rotating_hoop
x = np.linspace(-1, 1, 401)*3
y = np.linspace(-1, 1, 401)*2.5
rhodot, nudot = TrajectoryRepulsionRate().compute(x, y, f=bead_on_a_rotating_hoop, t=0, edge_order=2)
plt.pcolormesh(x, y, rhodot, shading='gouraud')
```
![alt text](https://github.com/hokiepete/docs/blob/main/images/bead_on_a_rotating_hoop_rhodot.png)
