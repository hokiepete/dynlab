import pytest
import numpy as np
from dynlab.flows import (
    double_gyre, autonomous_double_gyre, bickley_jet, glider, hills_vortex, sigmoidal,
    autonomous_duffing_oscillator, duffing_oscillator, pendulum, hurricane, van_der_pol_oscillator,
    bead_on_a_rotating_hoop, lotka_volterra, abc, lorenz
)


@pytest.fixture
def twodimcoordinates():
    x = [0, 1, 2]
    y = [0, 1]
    t = 0
    return t, *np.meshgrid(x, y)


def test_double_gyre(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = double_gyre(t, (x, y))
    u_expected = np.array([
        [-0.00000000e+00, -3.84734139e-17,  7.69468277e-17],
        [00.00000000e+00,  3.84734139e-17, -7.69468277e-17]
    ])
    v_expected = np.array([
        [0.00000000e+00, -0.00000000e+00, 0.00000000e+00],
        [3.84734139e-17, -3.84734139e-17, 3.84734139e-17]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_autonomous_double_gyre(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = autonomous_double_gyre(t, (x, y))
    u_expected = np.array([
        [-0.00000000e+00, -3.84734139e-17,  7.69468277e-17],
        [00.00000000e+00,  3.84734139e-17, -7.69468277e-17]
    ])
    v_expected = np.array([
        [0.00000000e+00, -0.00000000e+00, 0.00000000e+00],
        [3.84734139e-17, -3.84734139e-17, 3.84734139e-17]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_bickley_jet(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = bickley_jet(t, (x, y))
    u_expected = np.array([
        [5413.82400000, 5413.8240000, 5413.82400000],
        [5416.26919701, 5416.26919607, 5416.26919327]
    ])
    v_expected = np.array([
        [-0.0, -2.92741177, -5.85482113],
        [-0.0, -2.92741084, -5.85481926]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_glider(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = glider(t, (x, y))
    u_expected = np.array([
        [-0.00000000e+00, -4.00000000e-01, -1.60000000e+00],
        [01.46957616e-16, -2.82842712e-01, -1.43108351e+00]
    ])
    v_expected = np.array([
        [-1.0, -1.00000000, -1.0000000],
        [-3.4, -4.67695526, -7.0821049]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_hills_vortex(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = hills_vortex(t, (x, y))
    u_expected = np.array([
        [0., 0., 0.],
        [0., 2., 4.]
    ])
    v_expected = np.array([
        [2., -2., -14.],
        [0., -4., -16.]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_sigmoidal(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = sigmoidal(t, (x, y))
    u_expected = np.array([
        [0., 0.4, 0.4],
        [0., 0.4, 0.4]
    ])
    v_expected = np.array([
        [0, 0, 0],
        [0, 0, 0]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_autonomous_duffing_oscillator(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = autonomous_duffing_oscillator(t, (x, y))
    u_expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    v_expected = np.array([
        [0, 0, -6],
        [0, 0, -6]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_duffing_oscillator(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = duffing_oscillator(t, (x, y))
    u_expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    v_expected = np.array([
        [00.0,  0.0, -6.0],
        [-0.2, -0.2, -6.2]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_pendulum(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = pendulum(t, (x, y))
    u_expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    v_expected = np.array([
        [0.0, -0.84147098, -0.90929743],
        [0.0, -0.84147098, -0.90929743]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_hurricane(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = hurricane(t, (x, y))
    u_expected = np.array([
        [0.00000000e+00, -2.89929236e-01, -5.35185800e-01],
        [3.33066907e-16, -6.06384932e-01, -7.04965087e-01]
    ])
    v_expected = np.array([
        [0.0, 0.38898089, 0.35901355],
        [0.4, 1.21355076, 0.87290496]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_van_der_pol_oscillator(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = van_der_pol_oscillator(t, (x, y))
    u_expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    v_expected = np.array([
        [0.0, -1.0, -2.0],
        [1.9, -1.1, -8.1]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_bead_on_a_rotating_hoop(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = bead_on_a_rotating_hoop(t, (x, y))
    u_expected = np.array([
        [0, 0, 0],
        [1, 1, 1]
    ])
    v_expected = np.array([
        [000.0,  2.04221056, -17.79620296],
        [-10.0, -7.95778944, -27.79620296]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


def test_lotka_volterra(twodimcoordinates):
    t, x, y = twodimcoordinates
    u, v = lotka_volterra(t, (x, y))
    u_expected = np.array([
        [0.0,  0.66666667,  1.33333333],
        [0.0, -0.66666667, -1.33333333]
    ])
    v_expected = np.array([
        [00.,  0.,  0.],
        [-1.,  0.,  1.]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)


@pytest.fixture
def threedimcoordinates():
    x = [0, 1]
    y = [0, 1]
    z = [0, 1]
    t = 0
    return t, *np.meshgrid(x, y, z)


def test_abc(threedimcoordinates):
    t, x, y, z = threedimcoordinates
    u, v, w = abc(t, (x, y, z))
    u_expected = np.array([
        [
            [1.00000000, 2.4574705],
            [1.00000000, 2.4574705]
        ], [
            [0.54030231, 1.9977728],
            [0.54030231, 1.9977728]
        ]
    ])
    v_expected = np.array([
        [
            [1.73205081, 0.93583105],
            [2.92207049, 2.12585072]
        ], [
            [1.73205081, 0.93583105],
            [2.92207049, 2.12585072]
        ]
    ])

    w_expected = np.array([
        [
            [1.41421356, 1.41421356],
            [0.76410285, 0.76410285]
        ], [
            [2.25568455, 2.25568455],
            [1.60557383, 1.60557383]
        ]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)
    assert np.allclose(w_expected, w)


def test_lorenz(threedimcoordinates):
    t, x, y, z = threedimcoordinates
    u, v, w = lorenz(t, (x, y, z))
    u_expected = np.array([
        [
            [000.,   0.],
            [-10., -10.]
        ], [
            [10.,  10.],
            [00.,   0.]
        ]
    ])
    v_expected = np.array([
        [
            [00.,  0.],
            [28., 27.]
        ], [
            [-1., -1.],
            [27., 26.]
        ]
    ])
    w_expected = np.array([
        [
            [0., -2.66666667],
            [0., -2.66666667]
        ], [
            [0., -2.66666667],
            [1., -1.66666667]
        ]
    ])
    assert np.allclose(u_expected, u)
    assert np.allclose(v_expected, v)
    assert np.allclose(w_expected, w)
