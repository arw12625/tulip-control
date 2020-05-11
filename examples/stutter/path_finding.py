#!/usr/bin/env python
"""This is an example of defining a 2d kinematic system with obstacles and performing the stutter abstraction
"""

from __future__ import print_function

import logging

import numpy as np

from cvxopt import solvers

from tulip import spec, synth, hybrid
from polytope import box2poly
from polytope import qhull
from polytope import solvers
import polytope as pc
from tulip.abstract import prop2part, PropPreservingPartition
from tulip.abstract.plot import plot_partition

from matplotlib import pyplot as plt


from tulip.abstract.cont_stutter_abstraction import compute_stutter_abstraction, StutterAbstractionSettings, StutterPlotData, AbstractionType
from tulip.abstract.discretization import discretize

import pickle

solvers.default_solver = 'glpk'
solvers.options['msg_lev']='GLP_MSG_OFF'
logging.basicConfig(level=logging.WARNING)
show = True

abs_type = 'dual'
export_name = "data/dual_slant_small_1.obj"

# Continuous state space
#cont_state_space = box2poly([[-3, 3], [-2, 2]])
cont_state_space = box2poly([[0, 6], [-5, 0]])

# Continuous dynamics
# (continuous-state, discrete-time)
A = np.array([[1, 0], [0, 1]])
B = 0.340001 * np.array([[1, 0], [0, 1]])
#B = np.array([[0.7, -0.7], [0.7, 0.7]])/8
E = np.array([[0], [0]])

U = box2poly(np.array([[-1., 1.], [-1, 1]]))
#zero disturbance
W = box2poly(np.array([[0, 0]]))

# Construct the LTI system describing the dynamics
sys_dyn = hybrid.LtiSysDyn(A, B, E, None, U, W, cont_state_space)

'''
obstacles = [box2poly(np.array([[-2, -1], [-2, 0]])),
             box2poly(np.array([[-2, -1], [1, 2]])),
             box2poly(np.array([[-1, 0], [-1, 0]])),
             box2poly(np.array([[1, 2], [-1, 2]]))]
'''
obstacles = [qhull(np.array([[1,-1],[1,-2],[2,-4],[3,-4],[3,-2],[2,-1]])),
             qhull(np.array([[3,-2],[3,-4],[5,-1],[4,-1]])),
             qhull(np.array([[2,0],[3,-1],[4,0]])),
             qhull(np.array([[6,-1],[5,-3],[6,-3]])),
             qhull(np.array([[4,-4],[4,-5],[5,-5],[5,-4]]))]

obstacle_reg = pc.Region(list_poly=obstacles)

'''
init_reg = pc.Region(list_poly=[box2poly(np.array([[2, 3], [1, 2]]))])

target_reg = pc.Region(list_poly=[box2poly(np.array([[-3, -2], [-2, -1]]))])
'''
init_reg = pc.Region(list_poly=[box2poly(np.array([[0, 1], [-1, 0]]))])

target_reg = pc.Region(list_poly=[box2poly(np.array([[5,6], [-5,-4]]))])

# Define atomic propositions for relevant regions of state space
cont_props = {}
#cont_props['obs'] = obstacle_reg
cont_props['init'] = init_reg
cont_props['target'] = target_reg

#cont_state_space = cont_state_space.diff(obstacle_reg)
#print(cont_state_space)
for obs in obstacle_reg:
    cont_state_space = cont_state_space.diff(obs)

# Compute the proposition preserving partition of the continuous state space
cont_partition = prop2part(cont_state_space, cont_props)

if show:
    plt.ion()
    fig1 = plt.figure()
    ax = fig1.add_subplot(1,1,1)
    plot_partition(cont_partition, ax=ax)
    #plt.show()
    plt.pause(.01)
    plt.ioff()

stutter_settings = None
if abs_type == "stutter_bi":
    stutter_settings = StutterAbstractionSettings(
        backwards_horizon=25, abstraction_type=AbstractionType.STUTTER_BISIMULATION,
        min_cell_volume=0.125, max_iter=30000, init_data_size=1000, max_num_poly_per_reg=50)
elif abs_type == "bisim":
    stutter_settings = StutterAbstractionSettings(
        backwards_horizon=1, abstraction_type=AbstractionType.STUTTER_BISIMULATION,
        min_cell_volume=0.125, max_iter=30000, init_data_size=1000, max_num_poly_per_reg=50)
elif abs_type == "dual":
    stutter_settings = StutterAbstractionSettings(
        backwards_horizon=1, abstraction_type=AbstractionType.STUTTER_DUAL_SIMULATION,
        min_cell_volume=0.125, max_iter=30000, init_data_size=1000, max_num_poly_per_reg=50)

# Given dynamics & proposition-preserving partition, compute stutter bisimulation
plot_data = StutterPlotData(save_img=False, plot_every=10)

stutter_dynamics = compute_stutter_abstraction(
        cont_partition, sys_dyn, stutter_settings, plot_data, init_index_list=[1]
)
print(stutter_dynamics)

abs_pwa = stutter_dynamics

#export abstraction for later use
filehandler = open(export_name, 'wb')
pickle.dump(abs_pwa, filehandler)

# Visualize transitions in continuous domain (optional)

plot_partition(stutter_dynamics.ppp, stutter_dynamics.ts,
               stutter_dynamics.ppp2ts) if show else None
