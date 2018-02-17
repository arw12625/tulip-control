#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:01:55 2018

@author: zexiang
"""
import logging

import numpy as np

import __init__
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part
from dual_simu_cont import discretize_dual
from tulip.abstract.plot import plot_partition

from polytope import box2poly

from tulip.hybrid import LtiSysDyn, PwaSysDyn

# Problem parameters
input_bound = 1.0
uncertainty = 0.01

def subsys0():
    A = np.array([[1.1052, 0.], [ 0., 1.1052]])
    B = np.array([[1.1052, 0.], [ 0., 1.1052]])
    E = np.array([[1,0], [0,1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [0.5, 2.]])

    sys_dyn = LtiSysDyn(A, B, E, None, U, W, dom)
    #sys_dyn.plot()

    return sys_dyn

def subsys1():
    A = np.array([[0.9948, 0.], [0., 1.1052]])
    B = np.array([[-1.1052, 0.], [0., 1.1052]])
    E = np.array([[1, 0], [0, 1]])

    U = box2poly([[-1., 1.], [-1., 1.]])
    U.scale(input_bound)

    W = box2poly([[-1., 1.], [-1., 1.]])
    W.scale(uncertainty)

    dom = box2poly([[0., 3.], [0., 0.5]])

    sys_dyn = LtiSysDyn(A, B, E, None, U, W, dom)
    #sys_dyn.plot()

    return sys_dyn

subsystems = [subsys0(), subsys1()]

cont_state_space = box2poly([[0., 3.], [0., 2.]])
plotting = True

# Build piecewise affine system from its subsystems
sys_dyn = PwaSysDyn(subsystems, cont_state_space)
if plotting:
    ax = sys_dyn.plot()
    ax.figure.savefig('pwa_sys_dyn.pdf')

# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

# Compute the proposition preserving partition of the continuous state space
show = True
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None

# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize_dual(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.1, plotit=show
)

ts = disc_dynamics.ts
## Synthesize
#ctrl = synth.synthesize('omega', specs,
#                        sys=disc_dynamics.ts, ignore_sys_init=True)
#assert ctrl is not None, 'unrealizable'
#
#
## Generate a graphical representation of the controller for viewing
#if not ctrl.save('continuous.png'):
#    print(ctrl)