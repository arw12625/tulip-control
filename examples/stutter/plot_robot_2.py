"""
Construct and evaluate a robot task system
"""
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
#logging.getLogger('tulip').setLevel(logging.ERROR)
logger.setLevel(logging.DEBUG)

#from nose.tools import assert_raises

import matplotlib
# to avoid the need for using: ssh -X when running tests remotely
matplotlib.use('Agg')
from matplotlib import pyplot as plt

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
import math

pretty_names = {'none':'No abstraction', 'bisim':'Bisimulation', 'stutter':'Stutter Bisimulation'}

def sides_to_dims(lengths):
    return [(l,l) for l in lengths]

def sys_sizes(dims):
    return [(dim[0] * dim[1]) ** 2 for dim in dims]

def average(data):
    return sum(data) / len(data)

def load_grid_data(dims, num_rob, abs_mode, suffix):
    path = "examples/stutter/data/grid/dim"+str(dims[0])+"_"+str(dims[1])+"_rob"+str(num_rob)+"_abs_"+abs_mode+""+suffix+".json"
    with open(path) as f:
        data = json.load(f)
    return data


def abstract_time_reg_plot(fig, ax, dims, modes, suffix):
    avg_abs_times = {mode : [] for mode in modes}
    abs_sizes = {mode : [] for mode in modes}
    orig_sizes = sys_sizes(dims)

    ax.set_xlabel('Number of states in orignal system')
    ax.set_ylabel('Time to compute abstraction (s)')
    ax2 = ax.twinx()
    ax2.set_ylabel('Number of states in abstraction')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_xlim([10,1000])
    ax2.set_ylim([10,1000])

    for mode in modes:
        for dim in dims:
            data = load_grid_data(dim,2,mode,suffix)
            avg_abs_times[mode].append((average(data['abs_times'])))
            abs_sizes[mode].append((data['abs_ts_sizes'][0]))
            if not len(set(data['abs_ts_sizes'])) == 1:
                print("Different abstraction sizes")
        ax.plot(orig_sizes, avg_abs_times[mode], "--", label=(pretty_names[mode]+" Time"))
        ax2.plot(orig_sizes, abs_sizes[mode], "-", label=(pretty_names[mode]+" States"))
    #ax2.plot(lengths, [l ** (2 * 2) for l in lengths], "-", label=("Original States"))
    leg = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2)
    leg.set_title('Abstraction Computation Times and Sizes',prop={'size':'large'})
    #fig.legend(bbox_to_anchor=(0.65,0.85))
    #fig.tight_layout()


def synth_sim_plot(fig, ax, dims, modes, suffix):
    avg_synth_times = {mode : [] for mode in modes}
    avg_sim_times = {mode : [] for mode in modes}
    orig_sizes = sys_sizes(dims)

    ax.set_xlabel('Number of states in original system')
    ax.set_ylabel('Synthesis Time (s)')
    ax2 = ax.twinx()
    ax2.set_ylabel('Simulation Time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax2.set_yscale('log')
    ax.set_xlim([10,1000])
    ax.set_ylim([0.1,1.3])

    for mode in modes:
        for dim in dims:
            data = {}
            try:
                data = load_grid_data(dim,2,mode,suffix)
            except:
                break
            avg_synth_times[mode].append(average(data['synth_times']))
            avg_sim_times[mode].append(average(data['sim_times'])/1000)
        ax.plot(orig_sizes[:len(avg_synth_times[mode])], avg_synth_times[mode], "--", label=(pretty_names[mode]+" Synth Time"))
        ax2.plot(orig_sizes[:len(avg_sim_times[mode])], avg_sim_times[mode], "-", label=(pretty_names[mode]+" Sim Time"))
    leg = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.9), ncol=2)
    leg.set_title('Synthesis and Simulation Times',prop={'size':'large'})
    #fig.legend(bbox_to_anchor=(0.65,0.85))
    #fig.tight_layout()


if __name__ == '__main__':
    dims = sides_to_dims(range(3,6))
    dims.insert(0,(2,3))
    dims.insert(0,(2,2))

    abs_fig, abs_ax = plt.subplots()
    abstract_time_reg_plot(abs_fig, abs_ax, dims, ['bisim', 'stutter'], '_absonly')
    abs_fig.savefig('examples/stutter/fig/abs_plot.png', bbox_inches='tight')

    ss_fig, ss_ax = plt.subplots()
    synth_sim_plot(ss_fig, ss_ax, dims, ['bisim', 'stutter'], '')
    ss_fig.savefig('examples/stutter/fig/ss_plot.png', bbox_inches='tight')
