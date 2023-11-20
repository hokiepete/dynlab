import numpy as np
from dynlab.diagnostics import FTLE, AttractionRate, Rhodot
from dynlab.flows import double_gyre


def test_ftle():
    x = [0, 1, 2]
    y = [0, 1]

    expected_ftle = np.array([
        [0.04001625, 0.16099239, 0.000000],
        [0.00000000, 0.16099239, 0.226121]
    ])
    assert np.allclose(
        expected_ftle,
        FTLE(num_threads=1).compute(x, y, double_gyre, (2, 0))
    )


def test_attraction_rate_with_function():
    x = [0, 1, 2]
    y = [0, 1]
    t = 0
    expected_attraction_rate = np.array([
        [-3.84734139e-17, -5.44096237e-17, -9.08234100e-18],
        [-6.16297582e-33, -7.69468277e-17, -1.24502583e-16]
    ])

    expected_repulsion_rate = np.array([
        [3.84734139e-17, 5.44096237e-17, 1.62975996e-16],
        [7.69468277e-17, 6.16297582e-33, 4.75557549e-17]
    ])
    attraction_rate, repulsion_rate = AttractionRate().compute(x, y, f=double_gyre, t=t)
    assert np.allclose(expected_attraction_rate, attraction_rate)
    assert np.allclose(expected_repulsion_rate, repulsion_rate)


def test_attraction_rate_with_velocity():
    x = [0, 1, 2]
    y = [0, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))
    expected_attraction_rate = np.array([
        [-3.84734139e-17, -5.44096237e-17, -9.08234100e-18],
        [-6.16297582e-33, -7.69468277e-17, -1.24502583e-16]
    ])

    expected_repulsion_rate = np.array([
        [3.84734139e-17, 5.44096237e-17, 1.62975996e-16],
        [7.69468277e-17, 6.16297582e-33, 4.75557549e-17]
    ])
    attraction_rate, repulsion_rate = AttractionRate().compute(x, y, u=u, v=v)
    assert np.allclose(expected_attraction_rate, attraction_rate)
    assert np.allclose(expected_repulsion_rate, repulsion_rate)


def test_rhodot_with_function():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    t = 0
    expected_rhodot = np.array([
        [00.00000000e+00, 5.39936208e-02, -3.65903910e-17],
        [-1.02587879e-01, 2.55475137e-02, -8.62938239e-02]
    ])

    expected_nudot = np.array([
        [0.00000000e+00, -5.38736592e-02,  1.07867280e-01],
        [5.27940046e-03,  8.66723390e-17, -6.47203680e-02]
    ])
    rhodot, nudot = Rhodot().compute(x, y, f=double_gyre, t=t)
    assert np.allclose(expected_rhodot, rhodot)
    assert np.allclose(expected_nudot, nudot)


def test_rhodot_with_velocity():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))
    expected_rhodot = np.array([
        [00.00000000e+00, 5.39936208e-02, -3.65903910e-17],
        [-1.02587879e-01, 2.55475137e-02, -8.62938239e-02]
    ])

    expected_nudot = np.array([
        [0.00000000e+00, -5.38736592e-02,  1.07867280e-01],
        [5.27940046e-03,  8.66723390e-17, -6.47203680e-02]
    ])
    rhodot, nudot = Rhodot().compute(x, y, u=u, v=v)
    assert np.allclose(expected_rhodot, rhodot)
    assert np.allclose(expected_nudot, nudot)
