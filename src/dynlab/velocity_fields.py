# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 12:52:04 2018

This is a series of velocity fields for testing dynamical systems diagnostics.

@author: pnola
"""


import numpy as np

#domain is [0, 2]x[0, 1]
def double_gyre(t,Y,A = 0.1,w = 0.2*np.pi,e = 0.25):
    a = e*np.sin(w*t)
    b = 1-2*e*np.sin(w*t)
    f = a*Y[0]**2+b*Y[0]
    dfdx = 2*a*Y[0]+b    
    u =-np.pi*A*np.sin(np.pi*f)*np.cos(Y[1]*np.pi)    
    v = np.pi*A*np.cos(np.pi*f)*np.sin(Y[1]*np.pi)*dfdx
    return [u,v]

#domain is [0, 2]x[0, 1]
def autonomous_double_gyre(t,Y):
    u =-np.pi*np.sin(np.pi*Y[0])*np.cos(Y[1]*np.pi)    
    v = np.pi*np.cos(np.pi*Y[0])*np.sin(Y[1]*np.pi)
    return [u,v]

#!!!This is a version of the bickley jet flow which has been scaled down from
#[0, 20000]x[-4000,4000] to [0, 5]x[-1,1]!!!
def bickley_jet(t,Y):
    U0=62.66 #m/s
    U0 = 3.6*U0 #km/hr
    U0 = U0/4000*24 #1/day
    L = 1770.0 #kms
    L = L/4000 # no dim
    A2 = 0.1
    A3 = 0.3
    c2 = 0.205*U0
    c3 = 0.461*U0
    re = 6371.0 #earths radius, kms
    re = re/4000 #no dim
    k2 = 4/re
    k3 = 6/re
    u = U0*((1/np.cosh(Y[1]/L))**2) + 2*A3*U0*np.tanh(Y[1]/L)*((1/np.cosh(Y[1]/L))**2)*np.cos(k3*(Y[0]-c3*t)) + 2*A2*U0*np.tanh(Y[1]/L)*((1/np.cosh(Y[1]/L))**2)*np.cos(k2*(Y[0]-c2*t))
    v = -k3*A3*L*U0*((1/np.cosh(Y[1]/L))**2)*np.sin(k3*(Y[0]-c3*t)) - k2*A2*L*U0*((1/np.cosh(Y[1]/L))**2)*np.sin(k2*(Y[0]-c2*t))
    return [u,v]

def cell(t,Y):
    u =-0.1*np.cos(np.pi*Y[0]-np.pi/2.0)*np.sin(np.pi*Y[1]-np.pi/2.0)
    v = 0.1*np.sin(np.pi*Y[0]-np.pi/2.0)*np.cos(np.pi*Y[1]-np.pi/2.0)    
    return [u,v]


def glider(t,Y):
    u = np.sqrt(Y[0]**2 + Y[1]**2)*( -Y[1]*( 1.2*np.sin(-2*np.arctan2(Y[1],Y[0]))) - (1.4-np.cos(-2*np.arctan2(Y[1],Y[0])))*Y[0])
    v = np.sqrt(Y[0]**2 + Y[1]**2)*( Y[0]*( 1.2*np.sin(-2*np.arctan2(Y[1],Y[0])) ) - (1.4-np.cos(-2*np.arctan2(Y[1],Y[0])))*Y[1]) - 1
    return [u,v]

	
def hills_vortex(t,Y):
    u = 2.0*Y[0]*Y[1];
    v = -2.0*(2.0*Y[0]*Y[0]-1.0) - 2.0*Y[1]*Y[1]
    return [u,v]
  
def sigmoidal(t,Y):
    u = 0.4*np.tanh(10.0*(Y[0]-0.5*t))
    v = 0.0
    return [u,v]
  
def duffing_oscillator(t,Y):
    u = Y[1]
    v = Y[0] - Y[0]*Y[0]*Y[0]
    return [u,v]
  
def campagnola(t,Y,pendulum_amplitude=0.5):
    u = 0.0;
    if(Y[0]<pendulum_amplitude*t+5): 
        v = 1.0
    else:
        v = -1.0
    return [u,v]

def pendulum(t,Y,pendulum_amplitude=0.5):
    u = Y[1]
    v = -np.sin(Y[0])-pendulum_amplitude*Y[1]*np.sin(np.pi*t)
    return [u,v]

  
def hurricane(t,Y,eye_alpha=0.2,eye_omega=0.8,eye_amplitude=0.4):
    sqrtterm = np.sqrt(1.0+4.0*eye_alpha)
    beta = 1.0/sqrtterm
    yterm = Y[1]-0.5*(1.0+sqrtterm)
  
    u = -yterm/(Y[0]*Y[0]+yterm*yterm+eye_alpha) - beta
    v = Y[0]/(Y[0]*Y[0]+yterm*yterm+eye_alpha) + eye_amplitude*Y[1]*np.cos(eye_omega*t)
    return [u,v]
  
def wedge(t,Y,W_lambda=0.5,W_alpha=0.5,W_beta=0.5,W_K0=0.5):
  
    c1 = np.cos(W_lambda*Y[1])
    s1 = np.sin(W_lambda*Y[1])
  
    c2 = np.cos((W_lambda-2.0)*Y[1])
    s2 = np.sin((W_lambda-2.0)*Y[1])
  
    psi_theta = np.real(pow(Y[0],W_lambda)*W_K0*(-W_alpha*W_lambda*s1 + (W_lambda-2.0)*W_beta*s2))
    psi_r = np.real(W_lambda*pow(Y[0],W_lambda-1.0)*W_K0*(W_alpha*c1 - W_beta*c2))
  
    u = 2.0*np.pi*psi_theta/Y[0]
    v = -2.0*np.pi*psi_r/Y[0]
    return [u,v]

def weight(t,Y):
    weight = 0.1+np.exp(-(Y[0]*Y[0]+Y[1]*Y[1])/4.0)
    u = (-2*Y[1]-Y[1]*Y[1])*weight;
    v = (2*Y[0]-Y[0]*Y[1])*weight
    return [u,v]

def van_der_pol_oscillator(t,Y,epsilon=0.01,a=0.575):
    u = 1/epsilon*(Y[1]+Y[0]-Y[0]**3)
    v = a - Y[0]
    return [u,v]

def kevrekidis(t,Y, eps=0.01):
  return [-Y[0]-Y[1]+2, 1/eps*(Y[0]**3-Y[1])]

def ex11(t,Y):
  return [-(np.tanh(Y[0]**2/4)+Y[0]),-(Y[0]+2*Y[1])]

def rotHoop(t,Y, eps=0.1, gamma=2.3):
  return [Y[1], 1/eps*(np.sin(Y[0])*(gamma*np.cos(Y[0])-1)-Y[1])]


def verhulst(t,Y, eps=0.01):
  return [1, 1/eps*(Y[0]*Y[1]-Y[1]**2)]

def abc(t,Y, ABC_Amplitude=0.0):
  Ap = np.sqrt(3)
  Bp = np.sqrt(2)
  u = (Ap+ABC_Amplitude*np.sin(np.pi*t))*np.sin(Y[2]) + np.cos(Y[1])
  v = Bp*np.sin(Y[0]) + (Ap+ABC_Amplitude*np.sin(np.pi*t))*np.cos(Y[2])
  w = np.sin(Y[1]) + Bp*np.cos(Y[0])
  return [u,v,w]
