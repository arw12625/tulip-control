# !/usr/bin/env python
'''An example of using the stutter abstraction control lifting algorithm to control a linear system'''
import logging
from tulip.transys.transys import FTS, simu_abstract
from tulip.abstract.cont_stutter_abstraction import *
import numpy as np
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip import hybrid, spec, synth
import pickle

from matplotlib import pyplot as plt
import matplotlib.lines as mlines

import time

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_abstraction():
    '''Build the abstracted system and define a specification
    '''
    stutter_sys = pickle.load( open( "data/path_abs_bi_no_obs.obj", "rb" ) )
    orig_sys_dyn = stutter_sys.pwa

    # Environment variables and requirements
    env_vars = set()
    env_init = set()
    env_prog = set()
    env_safe = set()

    # System variables and requirements
    sys_vars = set()
    sys_init = {'init'}
    sys_prog = {'target'}
    #sys_safe = {'!obs'}
    sys_safe = set()
    sys_prog |= set()

    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                        env_safe, sys_safe, env_prog, sys_prog)
    specs.qinit = '\E \A'

    return stutter_sys, orig_sys_dyn, specs


def linear_stutter_control():

    start_time = time.time()

    # build an example finite transition system with a desired specification
    [stutter_ts, orig_sys_dyn, specs] = build_abstraction()

    plt.ion()
    fig1 = plt.figure()
    ax = fig1.add_subplot(1, 1, 1)
    plot_partition(stutter_ts.ppp, ax=ax)
    plt.pause(.01)
    plt.ioff()

    # synthesize a controller for the abstraction as a Mealy machine representing the closed loop dynamics
    ctrl_mealy = synth.synthesize(specs, sys=stutter_ts.ts, ignore_sys_init=False)
    print(ctrl_mealy)

    # identify an initial control and system state for simulation
    ctrl_state, _, edge_data = ctrl_mealy.edges('Sinit', data=True)[0]
    orig_reg = stutter_ts.ppp[edge_data['loc']]
    orig_state = orig_reg.chebXc

    # simulate the closed loop system for a fixed horizon and record the input and state sequence
    input_seq = []
    state_seq = []
    state_seq.append(orig_state)
    for i in range(25):
        print("step ", i)
        # Using the controller on the abstraction, determine an admissible sequence of regions
        admis_state_reg_seq = get_admis_from_stutter_ctrl(orig_state, orig_sys_dyn, stutter_ts, ctrl_state, ctrl_mealy, 0.0001, 100)
        for admis_state_reg in admis_state_reg_seq:
            print("cur_orig_state", orig_state)
            print("cur_ctrl_state ", ctrl_state)

            # Select an admissible input to drive the current state to the next admissible region
            admis_input_reg = _compute_admissible_input_region(orig_state, admis_state_reg, orig_sys_dyn)
            input_sel = admis_input_reg.chebXc

            # update the state of the original system using this input
            orig_state = orig_sys_dyn.A.dot(orig_state) + orig_sys_dyn.B.dot(input_sel)
            input_seq.append(input_sel)
            state_seq.append(orig_state)

            # Update the controller's state to match the selected next state
            ctrl_state = update_stutter_ctrl_state(ctrl_state, orig_state, stutter_ts, ctrl_mealy)

            print("orig_input", input_sel)
            print('next_orig_state', orig_state)
            print("next_ctrl_state ", ctrl_state)

    print("Path planning time %s[sec]" % (time.time() - start_time))

    # plot trajectory on partition
    for first, second in zip(state_seq, state_seq[1:]):
        l = mlines.Line2D([first[0], second[0]], [first[1], second[1]])
        ax.add_line(l)
    plt.show()

def _compute_admissible_input_region(state, target_reg, sys_dyn):
    '''Construct the region of admissible inputs by combining linear constraints'''
    input_reg = pc.Region()
    if not isinstance(target_reg, pc.Region):
        target_reg = pc.Region([target_reg])
    for polytope in target_reg:
        input_poly = pc.Polytope(
            A=np.concatenate((polytope.A.dot(sys_dyn.B), sys_dyn.Uset.A), axis=0),
            b=np.concatenate(
                (polytope.b - polytope.A.dot(sys_dyn.A).dot(state), sys_dyn.Uset.b), axis=0))
        input_poly = pc.reduce(input_poly)
        input_reg = input_reg.union(input_poly, check_convex=True)
    input_reg = pc.reduce(input_reg)

    return input_reg

if __name__ == "__main__":
    linear_stutter_control()