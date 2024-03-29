import numpy as np
from dynlab.diagnostics import (
    AttractionRate, RepulsionRate, TrajectoryRepulsionRate, TrajectoryRepulsionRatio, iLES
)
from dynlab.flows import double_gyre


def test_attraction_rate_with_function():
    x = [0, 1, 2]
    y = [0, 1]
    t = 0

    expected_attraction_rate = np.array([
        [-3.84734139e-17, -5.44096237e-17, -9.08234100e-18],
        [-6.16297582e-33, -7.69468277e-17, -1.24502583e-16]
    ])

    attraction_rate = AttractionRate().compute(x, y, f=double_gyre, t=t)

    assert np.allclose(expected_attraction_rate, attraction_rate)


def test_repulsion_rate_with_function():
    x = [0, 1, 2]
    y = [0, 1]
    t = 0

    expected_repulsion_rate = np.array([
        [3.84734139e-17, 5.44096237e-17, 1.62975996e-16],
        [7.69468277e-17, 6.16297582e-33, 4.75557549e-17]
    ])

    repulsion_rate = RepulsionRate().compute(x, y, f=double_gyre, t=t)

    assert np.allclose(expected_repulsion_rate, repulsion_rate)


def test_attraction_rate_with_velocity():
    x = [0, 1, 2]
    y = [0, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))

    expected_attraction_rate = np.array([
        [-3.84734139e-17, -5.44096237e-17, -9.08234100e-18],
        [-6.16297582e-33, -7.69468277e-17, -1.24502583e-16]
    ])

    attraction_rate = AttractionRate().compute(x, y, u=u, v=v)
    assert np.allclose(expected_attraction_rate, attraction_rate)


def test_repulsion_rate_with_velocity():
    x = [0, 1, 2]
    y = [0, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))

    expected_repulsion_rate = np.array([
        [3.84734139e-17, 5.44096237e-17, 1.62975996e-16],
        [7.69468277e-17, 6.16297582e-33, 4.75557549e-17]
    ])

    repulsion_rate = RepulsionRate().compute(x, y, u=u, v=v)

    assert np.allclose(expected_repulsion_rate, repulsion_rate)


def test_trajectory_repulsion_rate_with_function():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    t = 0

    expected_rhodot = np.array([
        [00.00000000e+00, 5.39936208e-02, -3.65903910e-17],
        [-1.02587879e-01, 2.55475137e-02, -8.62938239e-02]
    ])

    rhodot = TrajectoryRepulsionRate().compute(x, y, f=double_gyre, t=t)

    assert np.allclose(expected_rhodot, rhodot)


def test_trajectory_repulsion_ratio_with_function():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    t = 0

    expected_nudot = np.array([
        [0.00000000e+00, -5.38736592e-02,  1.07867280e-01],
        [5.27940046e-03,  8.66723390e-17, -6.47203680e-02]
    ])
    nudot = TrajectoryRepulsionRatio().compute(x, y, f=double_gyre, t=t)

    assert np.allclose(expected_nudot, nudot)


def test_trajectory_repulsion_rate_with_velocity():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))

    expected_rhodot = np.array([
        [00.00000000e+00, 5.39936208e-02, -3.65903910e-17],
        [-1.02587879e-01, 2.55475137e-02, -8.62938239e-02]
    ])

    rhodot = TrajectoryRepulsionRate().compute(x, y, u=u, v=v)

    assert np.allclose(expected_rhodot, rhodot)


def test_trajectory_repulsion_ratio_with_velocity():
    x = [0.1, 1, 2]
    y = [0.1, 1]
    u, v = double_gyre(0, np.meshgrid(x, y))

    expected_nudot = np.array([
        [0.00000000e+00, -5.38736592e-02,  1.07867280e-01],
        [5.27940046e-03,  8.66723390e-17, -6.47203680e-02]
    ])

    nudot = TrajectoryRepulsionRatio().compute(x, y, u=u, v=v)

    assert np.allclose(expected_nudot, nudot)


def test_iles():
    expected_attracting = [
        np.array([
            [0.00000000, 0.05593946],
            [0.03594183, 0.00000000]]),
        np.array([
            [2.00000000, 0.05593946],
            [1.96405817, 0.00000000]]),
        np.array([
            [1.00000000, 1.00000000],
            [1.00000000, 0.88888889],
            [1.00000000, 0.77777778],
            [1.00000000, 0.66666667],
            [1.00000000, 0.55555556]
        ])
    ]
    expected_repelling = [
        np.array([
            [0.00000000, 0.03799861],
            [0.05295883, 0.00000000]]),
        np.array([
            [1.00000000, 0.00000000],
            [1.00000000, 0.11111111],
            [1.00000000, 0.22222222],
            [1.00000000, 0.33333333],
            [1.00000000, 0.44444444]]),
        np.array([
            [1.94704117, 0.00000000],
            [2.00000000, 0.03799861]
        ])
    ]
    x = np.linspace(0, 2, 20)
    y = np.linspace(0, 1, 10)

    attracting_iles = iLES().compute(x, y, f=double_gyre, t=0, kind='attracting')
    repelling_iles = iLES().compute(x, y, f=double_gyre, t=0, kind='repelling')

    assert all([np.isclose(x, y).all() for x, y in zip(expected_attracting, attracting_iles)])
    assert all([np.isclose(x, y).all() for x, y in zip(expected_repelling, repelling_iles)])


def test_iles_with_forced_eigenvectors():
    expected_attracting = [np.array([
        [1.00000000, 0.55555556],
        [1.00000000, 0.66666667],
        [1.00000000, 0.77777778],
        [1.00000000, 0.88888889],
        [1.00000000, 1.00000000]
    ])]
    expected_repelling = [np.array([
        [1.00000000, 0.00000000],
        [1.00000000, 0.11111111],
        [1.00000000, 0.22222222],
        [1.00000000, 0.33333333],
        [1.00000000, 0.44444444]
    ])]
    x = np.linspace(0, 2, 20)
    y = np.linspace(0, 1, 10)

    attracting_iles = iLES().compute(
        x, y, f=double_gyre, t=0, kind='attracting', force_eigenvectors=True
    )
    repelling_iles = iLES().compute(
        x, y, f=double_gyre, t=0, kind='repelling', force_eigenvectors=True
    )

    assert all([np.isclose(x, y).all() for x, y in zip(expected_attracting, attracting_iles)])
    assert all([np.isclose(x, y).all() for x, y in zip(expected_repelling, repelling_iles)])
