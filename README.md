# dynlab

Dynlab is a python package to make dynamical systems analysis faster and easier, by providing practitioners prebuild modules for calculating common diagnostics. Dynlab currently offers several easy to use diagnostics for analyzing your dynamical systems including an FTLE calculator `FLTE`, an attraction and repulsion rate calculator `AttractionRate`, and a trajectory repulsion rate calculator `Rhodot`. Dynlab also offers users a variety of prewritten flows to experiment and play around with, including the double gyre `double_gyre`, the Duffing oscillator `duffing_oscillator`, the pendulum with damping and forcing `pendulum`, the Van Der Pol oscillator `van_der_pol_oscillator`, the Lotka-Voltera flow `lotka_volterra`, the Lorenz system `lorenz`, and many more. Note that all the flows are written in such a way that they can easily be passed to an ode integrator and as such even the autonomous flows will still require a time parameter to be passed to them.

This package is readily available on pypi and can be easily installed with the command `pip install dynlab`.
Note that dynlab is written with python 3.11 and you may need to update your python installation to take advantage of it.

# Example
```import numpy as np
import matplotlib.pyplot as plt
from dynlab.diagnostics import FTLE
from dynlab.flows import double_gyre
x = np.linspace(0, 2, 101)
y = np.linspace(0, 1, 101)
ftle = FTLE().compute(x, y, double_gyre, (10, 0), edge_order=2, rtol=1e-8, atol=1e-8)```
![alt text](https://github.com/hokiepete/docs/blob/main/double_gyre.png)
