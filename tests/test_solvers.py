import numpy as np
from dynlab.solvers import FTLESolver
from dynlab.velocity_fields import double_gyre

def test_ftle():
    expected_ftle = np.array([
        [0.0393375, 0.16070587, 0.        ],
        [0.       , 0.16070587, 0.22620488]
    ])
    x = [0,1,2]
    y = [0,1]
    assert np.allclose(
        expected_ftle,
        FTLESolver().compute(double_gyre, (2, 0), x, y)
    )