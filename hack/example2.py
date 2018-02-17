#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 17:27:51 2018

@author: zexiang
"""

import logging

import numpy as np
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition

# Problem parameters
input_bound = 1.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[0., 3.], [0., 2.]])

# Continuous dynamics
A = np.array([[1.0, 0.], [ 0., 1.0]])
B = np.array([[0.1, 0.], [ 0., 0.1]])
E = np.array([[1,0], [0,1]])

# Available control, possible disturbances
U = input_bound *np.array([[-1., 1.], [-1., 1.]])
W = uncertainty *np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)

# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['home'] = box2poly([[0., 1.], [0., 1.]])
cont_props['lot'] = box2poly([[2., 3.], [1., 2.]])

show = True
# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None

# Given dynamics & proposition-preserving partition, find feasible transitions
disc_dynamics = discretize(
    cont_partition, sys_dyn, closed_loop=True,
    N=8, min_cell_volume=0.1, plotit=show
)
