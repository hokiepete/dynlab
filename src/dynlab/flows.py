# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 12:52:04 2018

This is a series of velocity fields for testing dynamical systems diagnostics.

@author: pnola
"""

import numpy as np


def double_gyre(
    t: float, Y: tuple[float, float], A: float = 0.1, w: float = 0.2*np.pi, e: float = 0.25
) -> tuple[float, float]:
    """ time-dependent double gyre flow on the domain [0, 2]x[0, 1].
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            A (float): amplitude of the gyre velocity. Default is 0.1.
            w (float): frequency of the driving force. Default is 0.2*pi.
            e (float): strength of the driving force. Default is 0.25.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    a = e*np.sin(w*t)
    b = 1-2*e*np.sin(w*t)
    f = a*Y[0]**2+b*Y[0]
    dfdx = 2*a*Y[0]+b
    u = -np.pi*A*np.sin(np.pi*f)*np.cos(Y[1]*np.pi)
    v = np.pi*A*np.cos(np.pi*f)*np.sin(Y[1]*np.pi)*dfdx
    return u, v


def autonomous_double_gyre(_: float, Y: tuple[float, float]) -> tuple[float, float]:
    """ time-independent double gyre flow on the domain [0, 2]x[0, 1].
        Args:
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = -np.pi*np.sin(np.pi*Y[0])*np.cos(Y[1]*np.pi)
    v = np.pi*np.cos(np.pi*Y[0])*np.sin(Y[1]*np.pi)
    return u, v


def bickley_jet(
    t: float,
    Y: tuple[float, float],
    U0: float = 5413.824,  # km/day
    L: float = 1770.0,  # kms
    re: float = 6371.0,  # kms
    A2: float = 0.1,
    A3: float = 0.3,
    c2: float = 0.205,
    c3: float = 0.461,
    k2: float = 4.0,
    k3: float = 6.0,
    scale: float = 1.0
) -> tuple[float, float]:
    """ time-dependent bickley jet flow on the domain [0, 20000]x[-4000,4000].
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            U0 (float): the speed of the flow in km/day. Default is 5413.824.
            L (float): the canonical length of the flow. Default is 1770.0.
            re (float): earth's radius in kms. Default is 6371.0.
            A2 (float): weight to experiment with. Default is 0.1.
            A3 (float): weight to experiment with. Default is 0.3.
            c2 (float): weight to experiment with. Default is 0.205.
            c3 (float): weight to experiment with. Default is 0.461.
            k2 (float): weight to experiment with. Default is 4.0.
            k3 (float): weight to experiment with. Default is 6.0.
            scale (float): factor to scale down the domain of the flow, example scale=4000 will
                reduce the domain to [0, 5]x[-1, 1]. Default is 1.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    U0 /= scale
    L /= scale
    re /= scale
    c2 *= U0
    c3 *= U0
    k2 /= re
    k3 /= re
    u = U0*((1/np.cosh(Y[1]/L))**2) + 2*A3*U0*np.tanh(Y[1]/L)*((1/np.cosh(Y[1]/L))**2) \
        * np.cos(k3*(Y[0]-c3*t)) + 2*A2*U0*np.tanh(Y[1]/L)*((1/np.cosh(Y[1]/L))**2) \
        * np.cos(k2*(Y[0]-c2*t))
    v = -k3*A3*L*U0*((1/np.cosh(Y[1]/L))**2)*np.sin(k3*(Y[0]-c3*t)) - k2*A2*L*U0 \
        * ((1/np.cosh(Y[1]/L))**2)*np.sin(k2*(Y[0]-c2*t))
    return u, v


def glider(_: float, Y: tuple[float, float]) -> tuple[float, float]:
    """ time-independent glider flow.
        Args:
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = np.sqrt(Y[0]**2 + Y[1]**2)*(-Y[1]*(1.2*np.sin(-2*np.arctan2(Y[1], Y[0])))
                                    - (1.4-np.cos(-2*np.arctan2(Y[1], Y[0])))*Y[0])
    v = np.sqrt(Y[0]**2 + Y[1]**2)*(Y[0]*(1.2*np.sin(-2*np.arctan2(Y[1], Y[0])))
                                    - (1.4-np.cos(-2*np.arctan2(Y[1], Y[0])))*Y[1]) - 1
    return u, v


def hills_vortex(_: float, Y: tuple[float, float]) -> tuple[float, float]:
    """ time-independent hills vortex flow.
        Args:
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = 2.0*Y[0]*Y[1]
    v = -2.0*(2.0*Y[0]*Y[0]-1.0) - 2.0*Y[1]*Y[1]
    return u, v


def sigmoidal(t: float, Y: tuple[float, float]) -> tuple[float, float]:
    """ time-dependent sigmoidal flow.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = 0.4*np.tanh(10.0*(Y[0]-0.5*t))
    v = Y[1]*0  # ensures proper size when np.ndarrays are passed.
    return u, v


def autonomous_duffing_oscillator(_: float, Y: tuple[float, float]) -> tuple[float, float]:
    """ time-independent duffing oscillator flow.
        Args:
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = Y[1]
    v = Y[0] - Y[0]**3
    return u, v


def duffing_oscillator(
    t: float, Y: tuple[float, float], A: float = 0.3, gamma: float = 0.2, omega: float = 1.0
) -> tuple[float, float]:
    """ time-dependent duffing oscillator flow.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            A (float): is the strength of the driving force. Default is 0.3.
            gamma (float): is the coefficient of friction. Default is 0.2.
            omega (float): is the frequency of oscillation. Default is 1.0
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = Y[1]
    v = Y[0] - Y[0]**3 - gamma*Y[1] + A*np.sin(omega*t)
    return u, v


def pendulum(
    t: float, Y: tuple[float, float], A: float = 0.5, gamma: float = 0.0, omega: float = np.pi
) -> tuple[float, float]:
    """ time-dependent pendulum flow.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            A (float): amplitude of the forcing.
            omega (float): the frequency of the forcing.
            gamma (float): the damping coefficent.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = Y[1]
    v = -np.sin(Y[0]) - gamma*Y[1] + A*np.sin(omega*t)
    return u, v


def hurricane(
    t: float,
    Y: tuple[float, float],
    eye_alpha: float = 0.2,
    eye_omega: float = 0.8,
    eye_amplitude: float = 0.4
) -> tuple[float, float]:
    """ time-dependent flow simulating a hurricane centered on x=0, y=1.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            eye_alpha (float): weight to experiment with. Default is 0.2.
            eye_omega (float): frequency of eye rotation. Default is 0.8.
            eye_amplitude (float): amplitude of the eye. Default is 0.4
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    sqrtterm = np.sqrt(1.0+4.0*eye_alpha)
    beta = 1.0/sqrtterm
    yterm = Y[1]-0.5*(1.0+sqrtterm)
    u = -yterm / (Y[0]**2 + yterm**2 + eye_alpha) - beta
    v = Y[0] / (Y[0]**2 + yterm**2 + eye_alpha) + eye_amplitude*Y[1]*np.cos(eye_omega*t)
    return u, v


def van_der_pol_oscillator(
    t: float, Y: tuple[float, float], A: float = 0.2, mu: float = 2.0,
    gamma: float = 0.1, omega: float = np.pi
):
    """ time-dependent van der pol oscillator flow.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            A (float): amplitude of the forcing.
            omega (float): the frequency of the forcing.
            gamma (float): the damping coefficent.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = Y[1]
    v = mu*(1 - Y[0]**2)*Y[1] - Y[0] - gamma*Y[1] + A*np.sin(omega*t)
    return u, v


def bead_on_a_rotating_hoop(
    _: float, Y: tuple[float, float], epsilon: float = 0.1, gamma: float = 2.3
) -> tuple[float, float]:
    """ time-independent bead on a rotating hoop flow.
        Args
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            epsilon (float): a dimensionless parameter from the mass, radius, gravity, and damping
                of the system. A smaller eps equates to a larger damping. Default is 0.1.
            gamma (float): a dimensionless parameter from the radius, rotational speed, and gravity
                of the system. A smaller gamma equates to a slower hoop. Default is 2.3.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = Y[1]
    v = (1 / epsilon)*(np.sin(Y[0])*(gamma*np.cos(Y[0]) - 1) - Y[1])
    return u, v


def lotka_volterra(
    _: float, Y: tuple[float, float], alpha: float = 2.0/3.0,
    beta: float = 4.0/3.0, gamma: float = 1.0, delta: float = 1.0
):
    """ time-independent bead on a Lotka Volterra flow.
        Args
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y).
            alpha (float): maximum prey per capita growth rate. Default is 2/3.
            beta (float): the effect of predators on the prey growth rate. Default is 4/3.
            gamma (float): the predators per capita death rate. Default is 1.0.
            delta (float): the effect of prey on the predators growth rate. Default is 1.0.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
    """
    u = alpha*Y[0] - beta*Y[0]*Y[1]
    v = delta*Y[0]*Y[1] - gamma*Y[1]
    return u, v


def abc(
    t: float, Y: tuple[float, float, float], ABC_Amplitude: float = 0.0,
    A: float = np.sqrt(3), B: float = np.sqrt(2), C: float = 1.0
) -> tuple[float, float, float]:
    """ time-dependent Arnold-Beltrami-Childress flow.
        Args:
            t (float): the timestep of the flow.
            Y (tuple): the x and y coordinates of the point in the flow (x, y, z).
            ABC_Amplitude (float): amplitude of the forcing. Default is 0.0.
            A (float): parameter partially controlling flow amplitude in u and v.
            B (float): parameter partially controlling flow amplitude in v and w.
            C (float): parameter partially controlling flow amplitude in w and u.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
            w (float): the v velocity component.
    """
    u = (A+ABC_Amplitude*np.sin(np.pi*t))*np.sin(Y[2]) + C*np.cos(Y[1])
    v = B*np.sin(Y[0]) + (A+ABC_Amplitude*np.sin(np.pi*t))*np.cos(Y[2])
    w = C*np.sin(Y[1]) + B*np.cos(Y[0])
    return u, v, w


def lorenz(
    _: float, Y: tuple[float, float, float], sigma: float = 10.0,
    rho: float = 28.0, beta: float = 8.0/3.0
) -> tuple[float, float, float]:
    """ time-independent Lorenz system flow. NOTE: when using default parameters system will
            exhibit chaotic behavor and yield a strange attractor.
        Args
            _ (float): the timestep of the flow, not used but necessary for passing the flow to ODE
                integrators like scipy.integrate.solve_ivp.
            Y (tuple): the x and y coordinates of the point in the flow (x, y, z).
            sigma (float): tunable parameter. Default is 10.0.
            rho (float): tunable parameter. Default is 4/3.
            beta (float): tunable parameter. Default is 1.0.
        Returns:
            u (float): the u velocity component.
            v (float): the v velocity component.
            w (float): the v velocity component.
    """
    u = sigma*(Y[1] - Y[0])
    v = Y[0]*(rho - Y[2]) - Y[1]
    w = Y[0]*Y[1] - beta*Y[2]
    return u, v, w
