#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 07:50:55 2018

@author: zexiang
"""

from __future__ import print_function

import logging

import numpy as np
from tulip import spec, synth, hybrid
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip.abstract.plot import plot_partition
from dual_simu_cont import discretize_dual

from polytope import polytope as _pt

_pt.solver = 'mosek'

# @import_section_end@


logging.basicConfig(level=logging.WARNING)
show = False

# @dynamics_section@
# Problem parameters
input_bound = 1.0
uncertainty = 0.01

# Continuous state space
cont_state_space = box2poly([[-1, 1], [-1, 1]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[0.5, 1.0], [ 0.75, -1.0]])
B = np.array([[1, 0.], [ 0., 1]])
E = np.array([[0,0], [0,0]])

# Available control, possible disturbances
U = input_bound *np.array([[-1., 1.], [-1., 1.]])
W = uncertainty *np.array([[-1., 1.], [-1., 1.]])

# Convert to polyhedral representation
U = box2poly(U)
W = box2poly(W)

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['a'] = box2poly([[-0.5, 0.5], [-0.5, 0.5]])
cont_props['b'] = box2poly([[-1, -0.5], [-1, 1]])
cont_props['c'] = box2poly([[-0.5, 1], [0.5, 1]])
cont_props['d'] = box2poly([[0.5, 1], [-1, 0.5]])
cont_props['e'] = box2poly([[-0.5, 0.5], [-1, -0.5]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)
plot_partition(cont_partition) if show else None
# @partition_section_end@

# @discretize_section@
# Given dynamics & proposition-preserving partition, find feasible transitions
sys = discretize_dual(
    cont_partition, sys_dyn, closed_loop=True,
    trans_length=1,N=8, use_all_horizon=True, min_cell_volume=0.1, abs_tol=1e-7, plotit=show, simu_type='dual'
)
# @discretize_section_end@
plot_partition(sys.ppp, sys.ts,
               sys.ppp2ts) if True else None