"""
Construct and evaluate a robot task system
"""
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
#logging.getLogger('tulip').setLevel(logging.ERROR)
logger.setLevel(logging.DEBUG)

#from nose.tools import assert_raises

#import matplotlib
# to avoid the need for using: ssh -X when running tests remotely
#matplotlib.use('Agg')

import networkx as nx
import numpy as np

from tulip import transys as trs
from tulip.transys.algorithms import ts_sync_prod
from tulip import spec, synth
from tulip.abstract.finite_stutter_abstraction import simu_abstract_div_stutter, get_admis_from_stutter_ctrl, update_stutter_ctrl_state
from tulip.transys.transys import simu_abstract

#import sys
import time
import json
from datetime import date

def hypercube_graph(dimension, self_loops):
    # construct a graph whose nodes and edges are the vertices and edges of the hypercube with the given dimension
    g = nx.hypercube_graph(dimension)
    g = g.to_directed()
    if self_loops:
        for node in g.nodes():
            g.add_edge(node,node)
    return g

def grid2d_graph(dimension, self_loops):
    # construct a graph whose nodes and edges are the vertices and edges of the 2d grid with the given dimension
    g = nx.grid_graph(dimension)
    g = g.to_directed()
    if self_loops:
        for node in g.nodes():
            g.add_edge(node,node)
    return g

def construct_rob_sys(name, loc_graph, home_locs, task_locs):
    # construct a transition system representing a single robot moving on the given location graph

    ts = trs.FTS()
    ts.name = name

    if len(loc_graph) == 1:
        locs = [loc_graph.nodes()]
    else:
        locs = loc_graph.nodes()

    ts.states.add_from(locs)

    # we will define the initial states for the entire system later
    #ts.states.initial.add_from(ts.states())

    # The home ap for this robot is true if it is in a home position
    home_ap_name = 'home'+name
    ts.atomic_propositions.add(home_ap_name)
    for loc in home_locs:
        ts.states[loc]['ap'] = ts.states[loc]['ap'] | {home_ap_name}
    
    for transition in loc_graph.edges():
        ts.transitions.add(transition[0], transition[1])
    
    # The task i ap can be satisfied if this robot is in the position for task i
    for task in range(len(task_locs)):
        task_ap_name = 'task'+str(task)
        task_loc = task_locs[task]
        ts.atomic_propositions.add(task_ap_name)
        ts.states[task_loc]['ap'] = ts.states[task_loc]['ap'] | {task_ap_name}
    return ts
    
def construct_entire_sys(loc_graph, home_locs, task_locs, num_robots):
    # constuct a transition system by synchronously composing individual robots

    list_of_rob_ts = [construct_rob_sys(str(i), loc_graph, home_locs, task_locs) for i in range(num_robots)]
    
    # iteratively compute products
    ts = list_of_rob_ts[0]
    for i in range(1,num_robots):
        ts = ts_sync_prod(ts,list_of_rob_ts[i])
    
    ts.atomic_propositions.add('collision')
    
    for state in ts.states():
        loc_list = rob_state_to_locs(state, num_robots)
        if exists_collision(loc_list):
            # a collision occurs if two robots share the same position
            ts.states[state]['ap'] = ts.states[state]['ap'] | {'collision'}
        else:
            pass
            # initial states are those without collision
            ts.states.initial.add(state)
            #if 'home0' in ts.states[state]['ap'] and 'task1' in ts.states[state]['ap']:
            #    ts.states.initial.add(state)
        #ts.states.initial.add(state)

    # add a state with a self loop to ensure aperiodicity
    #test_state = next(iter([s for s,data in ts.states(data=True) if 'collision' not in data['ap']]))
    #ts.transitions.add(test_state, test_state)

    return ts

def construct_cube_sys(dimension, self_loops, num_robots):
    # Construct the entire system with robots moving on a hypercube
    assert dimension > 0
    loc_graph = hypercube_graph(dimension, self_loops)
    home_locs = [loc_graph.nodes()[0]]
    task_locs = [loc_graph.nodes()[-2]]
    return construct_entire_sys(loc_graph, home_locs, task_locs, num_robots), 1

def construct_grid2d_sys(dimension, self_loops, num_robots):
    # Construct the entire system with robots moving on a 2d grid
    loc_graph = grid2d_graph(dimension, self_loops)
    home_locs = [(0,0)]
    task_locs = [(0,dimension[1]-1), (dimension[0]-1,dimension[1]-1), (dimension[0]-1,0)]
    return construct_entire_sys(loc_graph, home_locs, task_locs, num_robots), 3

def locs_to_rob_state(loc_list):
    # map from list of robot locations to product system state
    state = loc_list[0]
    for i in range(1, len(loc_list)):
        state = (state, i)
    return state
    
def rob_state_to_locs(state, num_robots):
    # map from product system state to list of robot locations
    loc_list = []
    for i in range(num_robots-1):
        loc_list.append(state[1])
        state = state[0]
    loc_list.append(state)
    loc_list.reverse()
    return loc_list

def exists_collision(loc_list):
    # collision occurs when robot positions are not unique
    return len(loc_list) != len(set(loc_list))

def synthesize_controller(ts, specs):
    ctrl_mealy = synth.synthesize(specs, sys=ts, solver='gr1c')
    return ctrl_mealy

def construct_simple_specs(num_tasks, num_robots):
    # construct a spec ensuring no collisions, all tasks are completed infinitely often, and all robots visit home infinitely often

    # only states without collision are initial, so we can avoid the environment part of the GR(1) formula
    #env_init = {'!collision'}
    
    sys_prog = {'home'+str(rob) for rob in range(num_robots)} | {'task'+str(task) for task in range(num_tasks)}
    sys_safety = {'!collision'}

    specs = spec.GRSpec(sys_safety=sys_safety, sys_prog=sys_prog)
    specs.plus_one = False
    specs.qinit = '\A \E'
    specs.moore = False
    #specs.ignore_sys_init = True

    return specs

def hacky_int_ts(ts):
    # rename the states of the transition system to consecutive integers {0,1,2,...,n}
    int_ts = trs.FTS()
    ts_states = ts.states()
    for ap in ts.atomic_propositions:
        int_ts.atomic_propositions.add(ap)
    for i in range(len(ts_states)):
        int_ts.states.add(i, ap=ts.states[ts_states[i]]['ap'])
    for i in ts.states.initial:
        index = ts_states.index(i)
        int_ts.states.initial.add(index)
    for trans in ts.transitions():
        index0 = ts_states.index(trans[0])
        index1 = ts_states.index(trans[1])
        int_ts.transitions.add(index0,index1)
    return int_ts

def hacky_int_2_electric_boogaloo(ts,abs_ts,part):
    # rename the states of the abstracted transition system to consecutive integers and update the corresponding partition maps
    int_ts = trs.FTS()
    ts_states = abs_ts.states()
    new_num_states = 1 if len(ts_states) == 0 else 2**(len(ts_states) - 1).bit_length()
    for ap in abs_ts.atomic_propositions:
        int_ts.atomic_propositions.add(ap)
    for i in range(len(ts_states)):
        int_ts.states.add(i, ap=abs_ts.states[ts_states[i]]['ap'])
    for i in range(len(ts_states),new_num_states):
        # add states to ensure power of 2
        # this is necessary for some bdd things maybe?
        int_ts.states.add(i)
    for i in abs_ts.states.initial:
        index = ts_states.index(i)
        int_ts.states.initial.add(index)
    for trans in abs_ts.transitions():
        index0 = ts_states.index(trans[0])
        index1 = ts_states.index(trans[1])
        int_ts.transitions.add(index0,index1)
    part2 = {'ts2simu':{},'simu2ts':{}}
    for orig_state in ts.states():
        abs_state = part['ts2simu'][orig_state]
        part2['ts2simu'][orig_state] = set([ts_states.index(next(iter(abs_state)))])
    for state2 in range(len(ts_states)):
        abs_state = ts_states[state2]
        part2['simu2ts'][state2] = part['simu2ts'][abs_state]
    return int_ts, part2

def simple_ts():
    # define a simple transition system for debugging
    ts = trs.FTS()
    ts.atomic_propositions.add_from({'task0','home0','collision'})
    ts.states.add(0,ap={'home0'})
    ts.states.add(1,ap={'task0'})
    ts.states.initial.add(0)
    ts.states.initial.add(1)
    ts.transitions.add(0,0)
    ts.transitions.add(0,1)
    ts.transitions.add(1,0)
    ts.transitions.add(1,1)
    return ts, 1


def simulate_controller(ts, abs_ts, part, ctrl_mealy, init_ts_state, max_iter):
    # Simulate a path taken by the abstracted controller
    start_time = time.time()
    #print(abs_state)
    #print(part['simu2ts'])
    #print(list([data['loc'] for (source, dest, data) in ctrl_mealy.edges('Sinit', data=True)]))
    #print(abs_state)
    #print(ctrl_mealy.edges('Sinit',data=True))
    ctrl_state = 'Sinit'
    #ctrl_state = [dest for (source, dest, data) in ctrl_mealy.edges('Sinit', data=True) if data['loc'] == abs_state][0]
    #ctrl_state, edge_data = [(u,v) for (u,uvx,v) in ctrl_mealy.edges(data=True) if v['loc'] == abs_state][0]
    init_ts_state = next(iter(part['simu2ts'][next(iter([data['loc'] for _,_,data in ctrl_mealy.edges(ctrl_state,data=True)]))]))
    ts_state = init_ts_state
    abs_state = next(iter(part['ts2simu'][init_ts_state]))

    state_seq = [init_ts_state]
    
    for i in range(max_iter):
        admis_set = get_admis_from_stutter_ctrl(ts_state, ts, abs_ts, part, ctrl_state, ctrl_mealy)
        ts_state = next(iter(admis_set.intersection(ts.successors(ts_state))))
        ctrl_state = update_stutter_ctrl_state(ctrl_state, ts_state, part, ctrl_mealy)
        state_seq.append(ts_state)

    sim_time = time.time() - start_time
    
    return state_seq, sim_time

def construct_abstraction(ts, abs_mode):

    # abstract the transition system
    part_hash = {}
    start_time = time.time()
    if abs_mode == 'none':
        abs_ts = ts
        ts2abs = {s:set([s]) for s in ts.states()}
        abs2ts = ts2abs
        part_hash['ts2simu'] = ts2abs
        part_hash['simu2ts'] = abs2ts
    if abs_mode == 'stutter':
        abs_ts, part_hash = simu_abstract_div_stutter(ts)
    if abs_mode == 'bisim':
        abs_ts, part_hash = simu_abstract(ts, 'bi')
    abs_time = time.time() - start_time
    # fix abs_ts names
    abs_ts_2, part = hacky_int_2_electric_boogaloo(ts,abs_ts,part_hash)
    #print(abs_ts_2)

    return abs_ts_2, part, abs_time, len(abs_ts)


def perform_synth(abs_ts_2, specs):
    # synthesize controller for abstracted system
    abs_ts_2.owner = 'sys'
    start_time = time.time()
    ctrl = synthesize_controller(abs_ts_2, specs)
    synth_time = time.time() - start_time
    #print(ctrl)
    return ctrl, synth_time


if __name__ == '__main__':
    #We can increase the recursion limit that is encountered when parsing LTL, but this does not help very much as we usualy then run out of memory.
    #sys.setrecursionlimit(4000)
    
    map_type = 'grid'
    num_robots = 2
    dimension = [6,6]
    self_loops = True
    abs_mode = 'stutter' # stutter, bisim, or none
    num_tests = 25
    num_sim_steps = 1000
    params = {
        "map_type":map_type,
        "dimension":dimension,
        "num_robots":num_robots,
        "self_loops":self_loops,
        "abs_mode":abs_mode,
        "num_tests":num_tests,
        "num_sim_steps":num_sim_steps,
        "date":str(date.today())
    }
    test_name = "examples/stutter/data/"+map_type+"/dim"+str(dimension[0])+"_"+str(dimension[1])+"_rob"+str(num_robots)+"_abs_"+abs_mode+".json"

    test_data = {}

    if params['map_type'] == 'cube':
        ts, num_tasks = construct_cube_sys(dimension,self_loops,num_robots)
    elif params['map_type'] == 'grid':
        ts, num_tasks = construct_grid2d_sys(dimension,self_loops,num_robots)
    elif params['map_type'] == 'simple':
        ts, num_tasks = simple_ts()
    else:
        raise ValueError("Unknown map type")
    params["num_tasks"] = num_tasks
    params['ts_size'] = len(ts)

    # fix ts names
    ts = hacky_int_ts(ts)

    #init_state = ts.nodes()[0]
    # pick an initial state without collision
    init_state = [s for s in ts.states if 'collision' not in ts.states[s]['ap']][0]
    #print("INITIAL STATE: "+str(init_state))

    specs = construct_simple_specs(num_tasks, num_robots)
    #print(specs)

    params['abs_ts_sizes'] = []
    params['abs_times'] = []
    params['synth_times'] = []
    params['sim_times'] = []

    for i in range(params['num_tests']):
        abs_ts, part_hash, abs_time, abs_ts_size = construct_abstraction(ts, params['abs_mode'])
        params['abs_times'].append(abs_time)
        params['abs_ts_sizes'].append(abs_ts_size)

        ctrl, synth_time = perform_synth(abs_ts, specs)
        params['synth_times'].append(synth_time)

        state_seq, sim_time = simulate_controller(ts, abs_ts, part_hash, ctrl, init_state, params['num_sim_steps'])
        params['sim_times'].append(sim_time)

    with open(test_name, 'w') as outfile:
         json.dump(params, outfile)
