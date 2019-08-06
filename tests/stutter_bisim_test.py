
'''
Notes

Discretize can be modified slightly to compute the divergence stutter bisimulation for continuous systems 

Book suggests first must calculate divergent states. These are exactly the maximal controlled invariant sets of the elements of the proposition preserving partition.
However, I think this is only helpful for finite transition systems where divirgent sets must contain cycles, and there is no way to split a divirgent cycle with ppre.
This does not hold in the infinite case as there may exist infinite stutter paths that visit each of its states once.
That raises an interesting question though, can the divergent states of a polytopic region in a polytopic linear system be efficiently computed/described.

First modify solve_feasible to accept another argument restricting reachability analysis to only allow stutter steps.



'''#!/usr/bin/env python
"""Tests of `transys.transys.simu_abstract`."""
import logging
from tulip.transys.transys import FTS, simu_abstract
from tulip.abstract.finite_stutter_abstraction import *
import numpy as np
from polytope import box2poly
from tulip.abstract import prop2part, discretize
from tulip import hybrid, spec, synth

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_FTS():
    # build test FTS
    # simple test
    ts = FTS()
    ts.atomic_propositions.add_from({'a', 'b', 'c', 'd'})
    ts.states.add_from([('q1', {'ap': {'a'}}), ('q2', {'ap': {'a'}}),
                        ('q3', {'ap': {'b'}}), ('q4', {'ap': {'b'}}),
                        ('q5', {'ap': {'b'}}), ('q6', {'ap': {'c'}}),
                        ('q7', {'ap': {'d'}}), ('q8', {'ap': {'d'}}), 
                        ('q9', {'ap': {'a'}}), ('q10', {'ap': {'a'}})])
    ts.transitions.add('q1', 'q3')
    ts.transitions.add('q1', 'q4')
    ts.transitions.add('q3', 'q6')
    ts.transitions.add('q4', 'q6')
    ts.transitions.add('q2', 'q4')
    ts.transitions.add('q2', 'q5')
    ts.transitions.add('q5', 'q7')
    ts.transitions.add('q6', 'q6')
    ts.transitions.add('q7', 'q7')
    ts.transitions.add('q7', 'q8')
    ts.transitions.add('q8', 'q8')
    ts.transitions.add('q9', 'q2')
    ts.transitions.add('q10', 'q1')
    ts.transitions.add('q10', 'q10')
    ts.states.initial.add('q2')
    
    
    # Environment variables and requirements
    env_vars = set()
    env_init = set()           
    env_prog = set()
    env_safe = set()               

    # System variables and requirements
    sys_vars = set()
    sys_init = set()
    sys_prog = {'a || c || d'}               
    sys_safe = set()
    sys_prog |= set()

    specs = spec.GRSpec(env_vars, sys_vars, env_init, sys_init,
                    env_safe, sys_safe, env_prog, sys_prog)
    specs.qinit = '\E \A'
    
    return ts, specs


def simu_abstract_test():
    #build an example finite transition system with a desired specification
    [ts, specs] = build_FTS()
    #compute the coarsest divergence stutter bisimulation abstraction
    #
    [stutter_ts, stutter_part] = simu_abstract_div_stutter(ts)
    print(stutter_ts)
    print(stutter_part)
    
    #synthesize a controller for the abstraction as a Mealy machine representing the closed loop dynamics
    ctrl_mealy = synth.synthesize(specs, sys=stutter_ts, ignore_sys_init=False)
    print(ctrl_mealy)
    
    #identify an initial control and system state for simulation
    ctrl_state, _,edge_data = ctrl_mealy.edges('Sinit', data=True)[0]
    orig_state = next(iter(ts.states.initial.intersection(stutter_part['simu2ts'][edge_data['loc']])))
    print(ctrl_state)
    print(orig_state)
    
    #simulate the closed loop system for a fixed horizon 
    for i in range(5):
        #Using the controller on the abstraction, determine admissible next states in the original system
        admis_state = get_admis_from_stutter_ctrl(orig_state, ts, stutter_ts, stutter_part, ctrl_state, ctrl_mealy)
        #Select an available, admissible next state
        orig_state = next(iter(admis_state.intersection(ts.successors(orig_state))))
        #Update the controller's state to match the selected next state
        ctrl_state = update_stutter_ctrl_state(ctrl_state, orig_state, stutter_part, ctrl_mealy)
        
        print(ctrl_state)
        print(orig_state)
    

if __name__ == "__main__":
    simu_abstract_test()
