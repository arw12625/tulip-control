# Copyright (c) 2011-2016 by California Institute of Technology
# Copyright (c) 2016 by The Regents of the University of Michigan
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
# 3. Neither the name of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
"""
Algorithms related to stutter abstractions of continuous dynamics.

See Also
========
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import scipy.sparse as sp

logger = logging.getLogger(__name__)

import os
from copy import deepcopy

import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

from polytope.plot import plot_partition, plot_transition_arrow
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn

from .prop2partition import PropPreservingPartition
from .feasible import solve_feasible

from tulip.abstract.discretization import AbstractPwa

debug = False


class StutterAbstractionSettings:
    """Settings for the stutter abstraction algorithms

      - backwards_horizon: The maximum length of a stutter path considered
          Nominally, the algorithm splits sets by computation of the
          infinite backwards horizon PPre, but in this implementation
          the horizon is restricted to a finite number of steps specified
          by this attribute

          type: C{int}

      - min_cell_volume: The minimum volume of a region identified by the algorithm
          Abstracted state sets encountered by the algorithm with a smaller
          volume are considered to be empty and ignored

          type: C{float}

      - abs_tol: The tolerance in volume for two regions to be considered the same
          A subset 's1' of set 's2' is identified to be the same as 's2' if the
          difference in volume of the two is less than abs_tol

          type: C{float}

      - max_iter: The maximum number of iterations before prematurely terminating

          type: C{int}

      - init_data_size: The expected number of abstracted states to allocate data for initially

          type: C{int}

      - allow_resize: Whether or not to allow the data to be reallocated dynamically
            Currently unimplemented

          type: C{bool}

    """
    def __init__(self, backwards_horizon=10, min_cell_volume=1e-3, abs_tol=1e-7,
                 max_iter=1e4, init_data_size=1000, allow_resize=False):
        self.backwards_horizon = backwards_horizon
        self.min_cell_volume = min_cell_volume
        self.abs_tol = abs_tol
        self.max_iter = max_iter
        self.init_data_size = init_data_size
        self.allow_resize = allow_resize


class _StutterAbstractionData:
    """Helper class for running the stutter abstraction algorithms:

      - sys_dyn: The dynamics of the system the abstraction is being computed for
            Currently only linear systems are supported.

          type: C{LtiSysDyn}

      - orig_ppp: The proposition preserving partition of the original system
          type: C{PropPreservingPartition}

      - settings: The settings to be use by the abstraction algorithm

          type: C{StutterAbstractionSettings}

    """
    def __init__(self, sys_dyn, orig_ppp, settings):

        self.settings = settings
        self.sys_dyn = sys_dyn
        self.orig_ppp = orig_ppp

        self.data_size = 0
        # The current solution, consisting of abstract states which are regions of the original system
        self.sol = deepcopy(orig_ppp.regions)
        self.is_divergent = np.empty([0])
        self.transitions = np.empty([0, 0])
        self.sol2ppp = np.empty([0])
        self.index_pairs = np.empty([0, 0])

        # Set the maximum size of the partition and allocate the corresponding data structures
        self.set_data_size(settings.init_data_size)
        # Map the original partition elements to themselves in the original ppp
        self.sol2ppp[0:self.num_regions()] = np.arange(self.num_regions())

        self.progress = list()


    def num_regions(self):
        """Return the number of regions in the current solution

        @rtype: C{int}

        """
        return len(self.sol)

    def is_terminated(self):
        """Return whether or not the algorithm has terminated by exhausting all pairs

        @rtype: C{bool}

        """
        return np.sum(self.index_pairs[0:self.num_regions(), 0:self.num_regions()]) == 0

    def update_progress(self):
        """Update the progress variables of the algorithm
        Returns the current progress ratio

        @rtype: C{float}
        """
        progress_ratio = 1 - float(np.sum(self.index_pairs[0:self.num_regions(), 0:self.num_regions()])) / self.num_regions() ** 2
        self.progress.append(progress_ratio)
        return progress_ratio

    def set_data_size(self, data_size):
        """Set the amount of space allocated for abstracted states

        @param data_size: The number of states to allocate data for
        @type data_size: C{int}
        """
        if data_size < self.data_size:
            logger.warning('Cannot decrease maximum number of partition elements')
            return

        # Whether the abstracted states are divergent or not
        new_is_divergent = np.zeros((data_size,), dtype=bool)
        # Which original partition each abstracted state is contained in
        new_sol2ppp = np.zeros((data_size,), dtype=int)
        # Initialize output transitions
        # transitions[i,j] == 1 represents a transition from state j to state i
        # this is the transpose of the standard directed adjacency matrix of the graph corresponding to the system
        new_transitions = np.zeros([data_size, data_size], dtype=int)
        # Initialize matrix for pairs to check
        new_index_pairs = (np.ones([data_size, data_size], dtype=int) - np.eye(data_size, dtype=int))

        if self.data_size > 0:
            new_is_divergent[0:self.data_size] = self.is_divergent
            new_sol2ppp[0:self.data_size] = self.sol2ppp
            new_transitions[0:self.data_size, 0:self.data_size] = self.transitions
            new_index_pairs[0:self.data_size, 0:self.data_size] = self.index_pairs
        self.is_divergent = new_is_divergent
        self.transitions = new_transitions
        self.sol2ppp = new_sol2ppp
        self.index_pairs = new_index_pairs
        self.data_size = data_size

    def trim_solution(self):
        """Trim the allocated data to only what is needed for the current solution

        """
        self.transitions = self.transitions[0:self.num_regions(), 0:self.num_regions()]
        self.is_divergent = self.is_divergent[0:self.num_regions()]
        self.sol2ppp = self.sol2ppp[0:self.num_regions()]
        self.data_size = self.num_regions()


class StutterPlotData:

    file_extension = 'pdf'

    """A class encapsulating data and behavior for plotting the algorithm's results
        
      - save_img: Whether or not to save images
      
        type: C{bool}
        
      - plot_every: How often to plot intermediate results of the algorithm
      
        type: C{int}
    """
    def __init__(self, save_img=False, plot_every=1):
        #plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2)
        self.ax1.axis('scaled')
        self.ax2.axis('scaled')
        self.save_img = save_img
        self.plot_every = plot_every

    def plot_intermediate(self, stutter_data, init_reg_o, target_reg_o, iter_count):
        """Plot the intermediate results of the abstraction algorithm

        @param stutter_data: The data structure encapsulating the current state of the algorithm
        @type stutter_data: L{_StutterAbstractionData}

        @param init_reg_o: The initial region chosen in the current iteration
        @type init_reg_o: L{Region}

        @param target_reg_o: The target region chosen in the current iteration
        @type target_reg_o: L{Region}

        @param iter_count: The current iteration number
        @type iter_count: C{int}
        """
        if iter_count % self.plot_every != 0:
            return
        init_reg = deepcopy(init_reg_o)
        target_reg = deepcopy(target_reg_o)

        part = stutter_data.orig_ppp
        tmp_part = PropPreservingPartition(
            domain=part.domain,
            regions=stutter_data.sol,
            prop_regions=part.prop_regions
        )
        # plot pair under reachability check
        self.ax2.clear()
        init_reg.plot(ax=self.ax2, color='green')
        target_reg.plot(self.ax2, color='red', hatch='o', alpha=0.5)
        plot_transition_arrow(init_reg, target_reg, self.ax2)
        self.fig.canvas.draw()
        # plot partition
        self.ax1.clear()
        plot_partition(tmp_part, stutter_data.transitions.T, ax=self.ax1, color_seed=23)
        # plot dynamics
        stutter_data.sys_dyn.plot(self.ax1, show_domain=False)
        # plot hatched continuous propositions
        part.plot_props(self.ax1)
        self.fig.canvas.draw()
        # scale view based on domain,
        # not only the current polytopes si, sj
        l, u = part.domain.bounding_box
        self.ax2.set_xlim(l[0, 0], u[0, 0])
        self.ax2.set_ylim(l[1, 0], u[1, 0])
        if self.save_img:
            fname = 'movie' + str(iter_count).zfill(3)
            fname += '.' + self.file_extension
            self.fig.savefig(fname, dpi=250)
        plt.pause(1)

    def plot_final(self, progress):
        """Plot the final results of the algorithm and save if necessary

        @param progress: A list of the progress ratios
        @type progress: list of L{float}
        """
        if self.save_img:
            fig, ax = plt.subplots(1, 1)
            plt.plot(progress)
            ax.set_xlabel('iteration')
            ax.set_ylabel('progress ratio')
            ax.figure.savefig('progress.pdf')


def compute_stutter_abstraction(orig_ppp, sys_dyn,
            stutter_settings,
            plot_data=None, simu_type='bi'):
    """Perform a stutter abstraction algorithm on the given continuous system and partition
    Returns an abstracted system

    @param orig_ppp: The proposition preserving partition of the original system
    @type orig_ppp: L{PropPreservingPartition}

    @param sys_dyn: The continuous system dynamics to be abstracted
        Currently only linear dynamics are supported
    @type sys_dyn: L{LtiSysDyn}

    @param stutter_settings: The settings to be use by the abstraction algorithm
    @type stutter_settings: C{StutterAbstractionSettings}

    @param plot_data: Data for plotting the results of the algorithm
    @type plot_data: L{StutterPLotData}

    @param simu_type: A string specifying the algorithm to be performed
        Either 'bi' for stutter bisimulation or 'dual' for stutter dual simulation
        Currently only 'bi' is implemented
    @type simu_type: C{string}

    @rtype: L{AbstractPwa}
    """

    if simu_type == 'bi':

        stutter_data = _StutterAbstractionData(sys_dyn, orig_ppp, stutter_settings)
        abstract_pwa = _compute_stutter_bisim(stutter_data, plot_data)

    elif simu_type == 'dual':
        raise NotImplementedError
    else:
        raise ValueError(
            'Unknown simulation type: "{st}"'.format(
                st=simu_type))
    return abstract_pwa


def _compute_stutter_bisim(stutter_data, plot_data):
    """Perform the stutter bisimulation algorithm on the given continuous system and partition
    Returns an abstracted system

    @param stutter_data: The data structure encapsulating the current state of the algorithm
    @type stutter_data: L{_StutterAbstractionData}

    @param plot_data: Data for plotting the results of the algorithm
    @type plot_data: L{StutterPLotData}

    @rtype: L{AbstractPwa}
    """
    if isinstance(stutter_data.sys_dyn, PwaSysDyn):
        raise NotImplementedError

    start_time = os.times()[0]

    # Compute divergent subsets of original partition elements
    for index in range(stutter_data.num_regions()):
        _split_divergent(index, stutter_data)

    iter_count = 0

    # Do the abstraction
    while not stutter_data.is_terminated() and iter_count < stutter_data.settings.max_iter:
        ind = np.nonzero(stutter_data.index_pairs[0:stutter_data.num_regions(), 0:stutter_data.num_regions()])
        init_index = ind[1][0]
        target_index = ind[0][0]
        init_reg = stutter_data.sol[init_index]
        target_reg = stutter_data.sol[init_index]
        _check_pair(init_index, target_index, stutter_data)

        progress_ratio = stutter_data.update_progress()
        msg = '\t total # polytopes: {n_cells}\n'.format(n_cells=stutter_data.num_regions())
        msg += '\t progress ratio: {pr}\n'.format(pr=progress_ratio)
        logger.info(msg)
        iter_count += 1

        if not (plot_data is None):
            plot_data.plot_intermediate(stutter_data, init_reg, target_reg, iter_count)

    stutter_data.trim_solution()

    new_part = PropPreservingPartition(
        domain=stutter_data.orig_ppp.domain,
        regions=stutter_data.sol,
        prop_regions=stutter_data.orig_ppp.prop_regions
    )

    abs_ts = _build_stutter_abstr_FTS(new_part, stutter_data)

    param = {
        'N': stutter_data.settings.backwards_horizon,
        'min_cell_volume': stutter_data.settings.min_cell_volume,
    }

    end_time = os.times()[0]
    msg = 'Total abstraction time: {time}[sec]'.format(time=
                                                       end_time - start_time)
    print(msg)
    logger.info(msg)

    if not (plot_data is None):
        plot_data.plot_final(stutter_data.progress)

    return AbstractPwa(
        ppp=new_part,
        ts=abs_ts,
        ppp2ts=abs_ts.states,
        pwa=stutter_data.sys_dyn,
        pwa_ppp=stutter_data.orig_ppp,
        ppp2pwa=None,
        ppp2sys=None,
        ppp2orig=stutter_data.sol2ppp,
        disc_params=param
    )


def _compute_divergent(region, sys_dyn, min_cell_volume, abs_tol, max_iter=20):
    """Compute the divergent subset of a region with respect to a continuous system

    @param region: the region to compute the divergent subset of
    @type region: L{Region}

    @param sys_dyn: the system dynamics
    @type sys_dyn: L{LtiSysDyn}

    @param min_cell_volume: The minimum volume of a region identified by this method
    @type min_cell_volume: C{float}

    @param abs_tol: The tolerance in volume for two regions to be considered the same
    @type abs_tol: C{float}

    @param max_iter: The maximum number of iterations before prematurely terminating
    @type max_iter: C{int}

    @return: the divergent subset
    @rtype: L{Region}
    """
    # using one step pre
    s = region
    for n in range(max_iter):
        spre = solve_feasible(s, s, sys_dyn)
        vol_diff = s.volume - spre.volume
        s = spre
        if s.volume < min_cell_volume:
            s = pc.Polytope()
            break
        if vol_diff < abs_tol:
            break
        if n == max_iter - 1:
            logger.debug("Computation of divergent subset did not converge. Consider increasing max_iter")
    return s

def _split_divergent(index, data):
    """Split the given region into its divergent and non-divergent subsets and update the abstraction variables

    @param index: The index of the region in the current solution to split
    @type index: C{int}

    @param data: The data structure encapsulating the current state of the algorithm
    @type data: L{_StutterAbstractionData}

    """

    region = data.sol[index]
    reg_div = _compute_divergent(region, data.sys_dyn, data.settings.min_cell_volume, data.settings.abs_tol)

    if reg_div.volume > data.settings.min_cell_volume:
        div_diff = region.diff(reg_div)
        if div_diff.volume > data.settings.min_cell_volume:
            new_index = _split_state(index, reg_div, div_diff, data)
            data.is_divergent[index] = True
            data.is_divergent[new_index] = False
        else:
            data.is_divergent[index] = True
    else:
        data.is_divergent[index] = False

def _split_state(index, s1, s2, data):
    """Split an abstracted state into two specified parts and update the corresponding data

    @param index: The index of the state that is being split
    @type index: C{int}

    @param s1: The first half the region is split into
    @type s1: L{Region}

    @param s2: The second half the region is split into
    @type s2: L{Region}

    @param data: The data structure encapsulating the current state of the algorithm
    @type data: L{_StutterAbstractionData}

    @return: the index of the new state added by splitting
    @rtype: C{int}
    """
    orig_region = data.sol[index]
    # Make sure new areas are Regions and add proposition lists
    if not isinstance(s1, pc.Region):
        s1 = pc.Region([s1], orig_region.props)
    else:
        s1.props = orig_region.props.copy()

    if not isinstance(s2, pc.Region):
        s2 = pc.Region([s2], orig_region.props)
    else:
        s2.props = orig_region.props.copy()

    new_index = data.num_regions()
    data.sol[index] = s1
    data.sol.append(s2)
    data.is_divergent[new_index] = data.is_divergent[index]
    data.sol2ppp[new_index] = data.sol2ppp[index]

    data.transitions[:, new_index] = data.transitions[:, index]
    data.transitions[index, :] = 0

    data.index_pairs[index, 0:data.num_regions()] = 1
    data.index_pairs[new_index, 0:data.num_regions()] = 1
    data.index_pairs[0:data.num_regions(), index] = 1
    data.index_pairs[0:data.num_regions(), new_index] = 1
    data.index_pairs[index, index] = 0
    data.index_pairs[new_index, new_index] = 0

    return new_index

def _compute_ppre(init_reg, target_reg, sys_dyn, N, min_cell_vol):
    """Compute ppre for a given initial and target region

    @param init_reg: the initial region
    @type init_reg: L{Region}

    @param target_reg: the target region
    @type target_reg: L{Region}

    @param sys_dyn: The system dynamics
    @type sys_dyn: L{LtiSysDyn}

    @param N: the number of backwards steps to consider in computing PPre
    @type N: C{int}

    @param min_cell_vol: The minimum volume of a region identified by this method
    @type min_cell_vol: C{float}

    @return: the ppre of the given regions
    @rtype: L{Region}
    """
    ppre_reg = solve_feasible(init_reg, target_reg, sys_dyn, N, closed_loop=True, use_all_horizon=True, trans_set = init_reg)
    if ppre_reg.volume > min_cell_vol:
        return ppre_reg
    else:
        return pc.Polytope()

def _check_pair(init_index, target_index, data):
    """Check if the given pair splits states and update the algorithm data appropriately

    @param init_index: The index of the initial region
    @type init_index: C{int}

    @param target_index: The index of the target region
    @type target_index: C{int}

    @param data: The data structure encapsulating the current state of the algorithm
    @type data: L{_StutterAbstractionData}
    """
    # i,j swapped in discretize_overlap
    data.index_pairs[target_index, init_index] = 0
    init_reg = data.sol[init_index]
    target_reg = data.sol[target_index]

    ppre_reg = _compute_ppre(init_reg, target_reg, data.sys_dyn, data.settings.backwards_horizon, data.settings.min_cell_volume)

    if True:
        msg = '\n Working with partition cells: {i}, {j}'.format(i=init_index,
                                                                 j=target_index)
        logger.info(msg)
        msg = '\t{i} (#polytopes = {num}), and:\n'.format(i=init_index,
                                                          num=len(init_reg))
        msg += '\t{j} (#polytopes = {num})\n'.format(j=target_index,
                                                     num=len(target_reg))
        msg += '\t Computed reachable set S0 with volume: '
        msg += '{vol}\n'.format(vol=ppre_reg.volume)
        logger.debug(msg)

    ppre_vol = ppre_reg.volume
    diff_reg = init_reg.diff(ppre_reg)
    diff_vol = diff_reg.volume

    if ppre_vol <= data.settings.min_cell_volume:
        logger.warning('\t too small: si \cap Pre(sj), '
                       'so discard intersection')
    if ppre_vol <= data.settings.min_cell_volume and ppre_reg:
        logger.warning('\t discarded non-empty intersection: '
                       'consider reducing min_cell_volume')
    if diff_vol <= data.settings.min_cell_volume:
        logger.warning('\t too small: si \ Pre(sj), so not reached it')
    # We don't want our partitions to be smaller than the disturbance set
    # Could be a problem since cheby radius is calculated for smallest
    # convex polytope, so if we have a region we might throw away a good
    # cell.
    if (ppre_vol > data.settings.min_cell_volume) and (diff_vol > data.settings.min_cell_volume):

        diff_ind = _split_state(init_index, ppre_reg, diff_reg, data)

        data.transitions[target_index, init_index] = 1

        msg = ''
        if logger.getEffectiveLevel() <= logging.DEBUG:
            msg += '\t\n Adding states {i} and {j}'.format(i=init_index, j=diff_ind)
            msg += '\n'
            logger.debug(msg)

        if data.is_divergent[init_index]:
            _split_divergent(init_index, data)
            _split_divergent(diff_ind, data)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            msg = ''
            msg += '\n\n Updated trans: \n{trans}'.format(trans=
                                                          data.transitions)
            msg += '\n\n Updated index_pairs: \n{index_pairs}'.format(index_pairs=data.index_pairs)
            logger.debug(msg)
        logger.info('Divided region: {i}\n'.format(i=init_index))
    elif diff_vol < data.settings.abs_tol:
        logger.info('Found: {i} ---> {j}\n'.format(i=init_index, j=target_index))
        data.transitions[target_index, init_index] = 1
    else:
        if logger.level <= logging.DEBUG:
            msg = '\t Unreachable: {i} --X--> {j}\n'.format(i=init_index, j=target_index)
            msg += '\t\t diff vol: {vol_diff}\n'.format(vol_diff=diff_vol)
            msg += '\t\t intersect vol: {vol_S0}\n'.format(vol_S0=ppre_vol)
            logger.debug(msg)
        else:
            logger.info('\t unreachable\n')
        data.transitions[target_index, init_index] = 0

def _build_stutter_abstr_FTS(new_part, data):
    """Build a finite transition system from the abstracted partition

    @param new_part: The partition solution computed by the algorithm
    @type new_part: list of L{Region}

    @param data: The data structure encapsulating the current state of the algorithm
    @type data: L{_StutterAbstractionData}

    @return: The finite transition system for the abstraction
    @rtype: L{FTS}
    """
    # check completeness of adjacency matrix
    if debug:
        tmp_part = deepcopy(new_part)
        tmp_part.compute_adj()

    for index in range(len(new_part)):
        if data.is_divergent[index]:
            data.transitions[index, index] = 1

    # Generate transition system and add transitions
    ofts = trs.FTS()
    adj = sp.lil_matrix(data.transitions.T)
    n = adj.shape[0]
    ofts_states = range(n)
    ofts.states.add_from(ofts_states)
    ofts.transitions.add_adj(adj, ofts_states)
    # Decorate TS with state labels
    atomic_propositions = set(data.orig_ppp.prop_regions)
    ofts.atomic_propositions.add_from(atomic_propositions)
    for state, region in zip(ofts_states, data.sol):
        state_prop = region.props.copy()
        ofts.states.add(state, ap=state_prop)

    return ofts
