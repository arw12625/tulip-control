# Copyright (c) 2013-2015 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
"""Transition System Module"""
from __future__ import absolute_import
import logging
from collections import Iterable
from pprint import pformat
from tulip.transys.labeled_graphs import (
    LabeledDiGraph, str2singleton, prepend_with)
from tulip.transys.mathset import PowerSet, MathSet
import networkx as nx
from networkx import MultiDiGraph
import numpy as np
from itertools import chain
from tulip.transys import FTS

def simu_abstract_div_stutter(ts):
    """Compute the coarsest divergent stutter bisimulation abstraction for a Finite Transition System.

    @param ts: input finite transition system, the one you want to get
                    its abstraction.
    @type ts: L{FTS}

    @return: the abstraction, and the corresponding partition.
    @rtype: L{FTS}, C{dict}

    References
    ==========

    1. Baier, C; Katoen, J.P.
       "Principles of Model Checking"
       section 7.8.4 Stuter Bisimulation Quotienting
    """

    # Does this method require that the graph is weakly connected / nonterminal?

    [cond_G, cond_S0, div_state] = _expand_divergence_system(ts)

    # initialize blocks and set node attributes accordingly
    sol = []
    for ap in cond_S0:
        sol.append(cond_S0[ap])
        # nx.set_node_attributes(cond_G, 'block', {s:(len(sol) - 1) for s in cond_S0[ap]})

    n_cells = len(cond_S0)
    IJ = np.ones([n_cells, n_cells])
    transitions = np.zeros([n_cells, n_cells])
    while np.sum(IJ) > 0:
        # get i,j from IJ matrix, i--->j
        ind = np.nonzero(IJ)
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        if i == j:
            # need to handle case when i == j?
            continue
        si = sol[i]
        sj = sol[j]

        # direct predecessors of sj
        pre_j = _pre(cond_G, sj)

        '''
        marked_blocks = {cond_G.node[s]['block'] for s in pre_j}

        if si not in marked_blocks:
            continue
        '''
        # check if any direct predecessors of sj are contained in si
        # if none are found, then ppre(sj,si) is empty
        pre_isect = si.intersection(pre_j)
        if not pre_isect:
            continue

        # compute the states of si whose outgoing transitions leave si
        bottom_states = _compute_bottom(si, cond_G)

        # of those find the states which cannot transition into si or sj
        unmarked_bottom = bottom_states.difference(pre_j)

        # the absence of such states indicates that sj is reachable from all states of si
        if not unmarked_bottom:
            transitions[j, i] = 1
            continue

        # initialize ppre with the direct predecessors of sj in si
        ppre = pre_isect
        # initialize the complement of ppre with the bottom states of si not in sj
        ppre_comp = unmarked_bottom

        # subgraph corresponding to si to simplify successor calculations
        si_G = cond_G.subgraph(si)

        # traverse the non bottom states of si_G in a backwards topological order to compute ppre
        traversal_order = reversed(
            list(nx.topological_sort(cond_G.subgraph(si.difference(bottom_states).difference(pre_j)))))
        for s in traversal_order:
            if any(x in ppre for x in si_G.successors_iter(s)):
                ppre.add(s)
            else:
                ppre_comp.add(s)

        # add new block for ppre and replace current block si with ppre_comp
        sol.append(ppre)
        sol[i] = ppre_comp
        # update transition matrix
        transitions = np.pad(transitions, (0, 1), 'constant')
        transitions[n_cells, :] = 0
        transitions[:, n_cells] = transitions[:, i]
        transitions[j, n_cells] = 1
        transitions[i, :] = 0

        # update IJ matrix
        IJ = np.pad(IJ, (0, 1), 'constant', constant_values=1)
        IJ[i, :] = 1
        IJ[:, i] = 1
        n_cells += 1
        IJ = ((IJ - transitions) > 0).astype(int)
    return _output_fts_stutter(ts, cond_G, transitions, sol, div_state)

def _pre(graph, list_n):
    """Find union of predecessors of nodes in graph defined by C{MultiDiGraph}.

    @param graph: a graph structure corresponding to a FTS.
    @type graph: C{networkx.MultiDiGraph}
    @param list_n: list of nodes whose predecessors need to be returned
    @type list_n: list of nodes in C{graph}
    @return: set of predecessors of C{list_n}
    @rtype: C{set}
    """
    pre_set = set()
    for n in list_n:
        pre_set = pre_set.union(graph.predecessors(n))
    return pre_set

def _output_fts_stutter(ts, cond_G, transitions, sol, div_state):
    env_actions = [
        dict(name='env_actions',
             values=ts.env_actions,
             setter=True)]
    sys_actions = [
        dict(name='sys_actions',
             values=ts.sys_actions,
             setter=True)]
    ts_simu = FTS(env_actions, sys_actions)
    ts2simu = dict()
    simu2ts = dict()
    Part_hash = dict(ts2simu=ts2simu, simu2ts=simu2ts)

    div_index = sol.index({div_state})
    n_cells = len(sol) - 1
    S = list(range(n_cells + 1))
    S.remove(div_index)

    for i in S:
        simu2ts[i] = set().union(*[cond_G.node[s]['members'] for s in sol[i]])
        for j in simu2ts[i]:
            ts2simu[j] = {i}

    S_init = set()
    for i in ts.states.initial:
        S_init = S_init.union(ts2simu[i])
    ts_simu.states.add_from(S)
    ts_simu.states.initial.add_from(S_init)
    AP = ts.aps
    ts_simu.atomic_propositions.add_from(AP)

    for i in S:
        ts_simu.states.add(i, ap=ts.node[next(iter(cond_G.node[next(iter(sol[i]))]['members']))]['ap'])
    for i in S:
        for j in range(n_cells):
            if transitions[j, i]:
                if j == div_index:
                    ts_simu.transitions.add(i, i)
                else:
                    ts_simu.transitions.add(i, j)

    return ts_simu, Part_hash


def _compute_bottom(region, ts_G):
    '''Compute the bottom of a region in a finite transition system
        A state is in the bottom of the region if all outgoing transitions exit the region

    '''
    bottom = set()
    for node in region:
        if not any([succ in region for succ in ts_G.successors_iter(node)]):
            bottom.add(node)

    return bottom


def _compute_stutter_components(region, ts_G):
    '''Compute the stutter components within a given region in the finite transition system
        For finite transition systems, divergent states are those that can reach stutter cycles
        via a stutter path. The union of stutter cycles connected by stutter paths are exactly
        the strongly connected components of the subgraph generated by the region which are
        called stutter components here.
    '''

    return nx.strongly_connected_components(ts_G.subgraph(region))


def _expand_divergence_system(ts):
    '''Simplify the given transition system for use with stutter quotienting
        Identify the stutter components for each region in the system
        and create a new transition system by condensing these components
        and adding adding a transition to a dummy state 'div' representing divergence
        The resulting system then has no divergent states

    @return: the expanded transition graph, the coarsest proposition preserving partition for the expanded system
    @rtype: C{MultiDiGraph}, C{dict}
    '''

    # create MultiDiGraph instance from the input FTS
    G = MultiDiGraph(ts)
    # build coarsest partition
    S0 = dict()
    n_cells = 0
    for node in G:
        ap = repr(G.node[node]['ap'])
        if ap not in S0:
            S0[ap] = set()
            n_cells += 1
        S0[ap].add(node)

    # Build list of stutter components
    cond_stutter_states = iter(())
    for ap in S0:
        cond_stutter_states = chain(cond_stutter_states, _compute_stutter_components(S0[ap], G))
    # Condense the transitions stutter components
    cond_G = nx.condensation(G, cond_stutter_states)

    # compute which condensed states correspond to divergent states
    div_cond_states = set()
    for s in cond_G:
        add_trans = False
        if len(cond_G.node[s]['members']) > 1:
            div_cond_states.add(s)
        else:
            orig_s = next(iter(cond_G.node[s]['members']))
            if orig_s in ts.successors(orig_s):
                div_cond_states.add(s)

    # Build coarsest partition on condensed system
    cond_S0 = dict()
    for ap in S0:
        cond_S0[ap] = set()
    for node in cond_G:
        ap = repr(ts.node[next(iter(cond_G.node[node]['members']))]['ap'])
        cond_S0[ap].add(node)

    # should check if div is already used
    div_state = 'div'
    div_ap = 'div_ap'
    cond_G.add_node(div_state)
    cond_S0[div_ap] = {div_state}
    for s in div_cond_states:
        cond_G.add_edge(s, div_state)

    return cond_G, cond_S0, div_state


def get_admis_from_stutter_ctrl(orig_state, orig_ts, stutter_ts, stutter_part, ctrl_state, ctrl_ts):
    '''Compute admissible state transitions in a transition system given a controller on a system resulting from quotienting by a divergent stutter bisimulation
    The controller must be specified as a finite transition system corresponding to the system's total behavior.
    Each controller state must correspond to one stutter abstraction state specified by an attribute on incoming transitions ('loc')
    Such a controller can be generated by the method C{synthesize}

    @param orig_state: the current state of the original transition system

    @param orig_ts: the original transition system to be controlled
    @type orig_ts L{FTS}

    @param stutter_ts: the divergent stutter bisimulation quotient system
    @type stutter_ts: L{FTS}

    @param stutter_part: mappings between the states of the original system and the quotient
    @type stutter_part: C{dict}

    @param ctrl_state: the current state of the controller system

    @param ctrl_ts: the controller system
    @type ctrl_ts: L{FTS}

    @return: the set of states admissible by the controller abstraction
    @rtype: C{set}
    '''
    # could improve this method by returning an admissible sequence instead of a single step

    # The state in the quotient corresponding to the current state
    stutter_state = next(iter(stutter_part['ts2simu'][orig_state]))

    # Whether this state is divergent in the quotient
    # Does this computation assume that the quotient is by the coarsest relation
    divergent = stutter_state in stutter_ts.successors(stutter_state)

    # The set of quotient states that are admissible by the controller
    # stutter_succ_set = {stutter_succ for _, _, stutter_succ in ctrl_ts.edges([ctrl_state], data='loc')}
    stutter_succ_set = {ctrl_ts.edges([ctrl_succ], data='loc')[0][2] for ctrl_succ in ctrl_ts.successors(ctrl_state)}
    orig_succ_set = set.union(*[stutter_part['simu2ts'][s] for s in stutter_succ_set])
    admis = set()
    if not divergent:
        admis = (stutter_part['simu2ts'][stutter_state]).union(orig_succ_set)
    else:
        if stutter_state in stutter_succ_set:
            admis = set.union(orig_succ_set)
        else:
            # States in the equivalence class of the current state that can transition in one step into an admissible equivalence class
            exit_states = set.union(*[orig_ts.predecessors(x) for x in orig_succ_set]).intersection(
                stutter_part['simu2ts'][stutter_state])
            if orig_state in exit_states:
                admis = set.union(orig_succ_set)
            else:
                cur_set = exit_states;
                old_set = cur_set;
                stutter_graph = MultiDiGraph(orig_ts).subgraph(stutter_part['simu2ts'][stutter_state])
                reached = False
                # iterate ppre until the original state is found
                while orig_state not in cur_set:
                    old_set = cur_set
                    cur_set = set.union(*[stutter_graph.predecessors(x) for x in cur_set])
                admis = old_set

    return admis


def update_stutter_ctrl_state(cur_ctrl_state, next_orig_state, stutter_part, ctrl_ts):
    '''Determine the controller state corresponding to a transition in the original system given the current controller state

    @param cur_ctrl_state: the current state of the controller system

    @param next_orig_state: the state of the original system that is being transitioned to

    @param stutter_part: mappings between the states of the original system and the quotient
    @type stutter_part: C{dict}

    @param ctrl_ts: the controller system
    @type ctrl_ts: L{FTS}

    @return: new controller state
    '''
    next_stutter_state = next(iter(stutter_part['ts2simu'][next_orig_state]))

    for next_state, _, corr_stutter_state in {ctrl_ts.edges([ctrl_succ], data='loc')[0] for ctrl_succ in
                                              ctrl_ts.successors(cur_ctrl_state)}:
        if corr_stutter_state == next_stutter_state:
            next_ctrl_state = next_state
            break

    return next_ctrl_state
