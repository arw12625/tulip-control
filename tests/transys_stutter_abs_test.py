#!/usr/bin/env python
"""Testing the stutter bisimulation abstraction algorithm on linear systems
"""
#This code is based of of 'continuous.py' which can be found in the examples folder

# @import_section@
from __future__ import print_function

import logging

import numpy as np
from tulip import spec, synth, hybrid
from polytope import box2poly
import polytope as pc
from tulip.abstract import prop2part, PropPreservingPartition
from tulip.abstract.plot import plot_partition

from tulip.abstract.cont_stutter_abstraction import compute_stutter_abstraction, StutterAbstractionSettings, StutterPlotData, AbstractionType
# @import_section_end@


logging.basicConfig(level=logging.WARNING)
show = True

# @dynamics_section@

'''
#one-dimensional example
# Continuous state space
cont_state_space = box2poly([[-1.5, 1.5]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[2]])
B = np.array([[1]])
E = np.array([[0]])

U = box2poly(np.array([[-2., 2.]]))
W = box2poly(np.array([[0, 0]]))

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['s1'] = box2poly([[-1.5, -1]])
cont_props['s2'] = box2poly([[-1, 1]])
cont_props['s3'] = box2poly([[1, 1.5]])
'''

#two-dimensional example
#taken from the dual simulation paper by Wagenmaker and Ozay
# Continuous state space
cont_state_space = box2poly([[-1, 1], [-1, 1]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[0.5, 1], [0.75, -1]])
B = np.array([[1, 0], [0, 1]])
E = np.array([[0], [0]])

U = box2poly(np.array([[-1., 1.], [-1, 1]]))
#zero disturbance
W = box2poly(np.array([[0, 0]]))

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)
# @dynamics_section_end@

# @partition_section@
# Define atomic propositions for relevant regions of state space
cont_props = {}
cont_props['s1'] = box2poly([[-0.5, 0.5], [-0.5, 0.5]])
cont_props['s2'] = box2poly([[-1, -0.5], [-1, 1]])
cont_props['s3'] = box2poly([[-0.5, 1], [0.5, 1]])
cont_props['s4'] = box2poly([[0.5, 1], [-1, 0.5]])
cont_props['s5'] = box2poly([[-0.5, 0.5], [-1, -0.5]])

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)

plot_partition(cont_partition) if show else None
# @partition_section_end@

# @abstraction_section@
# Given dynamics & proposition-preserving partition, compute stutter bisimulation

stutter_settings = StutterAbstractionSettings(backwards_horizon=5, abstraction_type=AbstractionType.STUTTER_BISIMULATION)
plot_data = StutterPlotData(save_img=False, plot_every=100)

stutter_dynamics = compute_stutter_abstraction(
        cont_partition, sys_dyn, stutter_settings, plot_data
)


print(stutter_dynamics)
# @abstraction_section@

# Visualize transitions in continuous domain (optional)

plot_partition(stutter_dynamics.ppp, stutter_dynamics.ts,
               stutter_dynamics.ppp2ts) if show else None

