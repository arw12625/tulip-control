# !/usr/bin/env python
'''An example of using the stutter abstraction control lifting algorithm to control a linear system'''

import logging
from tulip.transys.transys import FTS, simu_abstract
from tulip.abstract.cont_stutter_abstraction import *
import numpy as np
from tulip.abstract import prop2part, discretize
from tulip import spec, synth
import pickle

import time

logging.basicConfig()
logger = logging.getLogger(__name__)

def discrete_stutter_control(ts, abs_ts, ctrl_mealy, spec):

    start_time = time.time()

    # identify an initial control and system state for simulation
    ctrl_state, _, edge_data = ctrl_mealy.edges(0, data=True)[0]
    orig_reg = stutter_ts.ppp[edge_data['loc']]
    orig_state = np.array([0.5,-0.5001])#orig_reg.chebXc - [0,0.001]

    # simulate the closed loop system for a fixed horizon and record the input and state sequence
    input_seq = []
    state_seq = []
    state_seq.append(orig_state)
    for i in range(50):
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
    dist = 0
    x = [state_seq[ind][0] for ind in range(len(state_seq))]
    y = [state_seq[ind][1] for ind in range(len(state_seq))]
    ax.plot(x, y, 'k', linewidth=4, marker='o', markersize=8)
    for first, second in zip(state_seq, state_seq[1:]):
        #l = mlines.Line2D([first[0], second[0]], [first[1], second[1]])
        #ax.add_line(l,'k',linewidth=2)
        #ax.plot(first[0],first[1],'k',marker='o',markersize=8)
        dist = dist + np.sqrt((second[0] - first[0]) * (second[0] - first[0]) + (second[1] - first[1]) * (second[1] - first[1]))
    print(dist)
    plt.show()

def _compute_admissible_input_region(state, target_reg, sys_dyn):
    '''Construct the region of admissible inputs by combining linear constraints'''

    input_polys = []
    if not isinstance(target_reg, pc.Region):
        target_reg = pc.Region([target_reg])
    for polytope in target_reg:
        input_poly = pc.Polytope(
            A=np.concatenate((polytope.A.dot(sys_dyn.B), sys_dyn.Uset.A), axis=0),
            b=np.concatenate(
                (polytope.b - polytope.A.dot(sys_dyn.A).dot(state), sys_dyn.Uset.b*1.02), axis=0))
        input_polys.append(input_poly)
    input_reg = pc.Region(list_poly=input_polys)
    input_reg = pc.reduce(input_reg)

    return input_reg

if __name__ == "__main__":
    linear_stutter_control()
