#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 14:01:38 2018

@author: zexiang
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
logger = logging.getLogger(__name__)

import os
from copy import deepcopy

import numpy as np
from scipy import sparse as sp
import polytope as pc

from polytope.plot import plot_partition, plot_transition_arrow
from tulip import transys as trs
from tulip.hybrid import LtiSysDyn, PwaSysDyn
from tulip.abstract.discretization import reachable_within,solve_feasible,sym_adj_change,AbstractPwa

from tulip.abstract.prop2partition import (PropPreservingPartition,
                             pwa_partition, part2convex)

debug = False

def discretize_dual(
    part, ssys, N=10, min_cell_volume=0.1,
    closed_loop=True, conservative=False,
    max_num_poly=5, use_all_horizon=False,
    trans_length=1, remove_trans=False,
    abs_tol=1e-7,
    plotit=False, save_img=False, cont_props=None,
    plot_every=1, simu_type = 'bi'
):
    """Refine the partition and establish transitions
    based on reachability analysis.

    Reference
    =========
    U{[NOTM12]
    <https://tulip-control.sourceforge.io/doc/bibliography.html#notm12>}

    See Also
    ========
    L{prop2partition.pwa_partition}, L{prop2partition.part2convex}

    @param part: L{PropPreservingPartition} object
    @param ssys: L{LtiSysDyn} or L{PwaSysDyn} object
    @param N: horizon length
    @param min_cell_volume: the minimum volume of cells in the resulting
        partition.
    @param closed_loop: boolean indicating whether the `closed loop`
        algorithm should be used. default True.
    @param conservative: if true, force sequence in reachability analysis
        to stay inside starting cell. If false, safety
        is ensured by keeping the sequence inside a convexified
        version of the original proposition preserving cell.
    @param max_num_poly: maximum number of polytopes in a region to use in
        reachability analysis.
    @param use_all_horizon: in closed loop algorithm: if we should look
        for reachability also in less than N steps.
    @param trans_length: the number of polytopes allowed to cross in a
        transition.  a value of 1 checks transitions
        only between neighbors, a value of 2 checks
        neighbors of neighbors and so on.
    @param remove_trans: if True, remove found transitions between
        non-neighbors.
    @param abs_tol: maximum volume for an "empty" polytope

    @param plotit: plot partitioning as it evolves
    @type plotit: boolean,
        default = False

    @param save_img: save snapshots of partitioning to PDF files,
        requires plotit=True
    @type save_img: boolean,
        default = False

    @param cont_props: continuous propositions to plot
    @type cont_props: list of C{Polytope}
    
    @param simu_type: flag used to choose abstraction algorithm
        (bisimulation or dual-simulation). 
    @type simu_type: string, 'bi' or 'dual'
        default = 'bi'
    
    @rtype: L{AbstractPwa}
    """
    start_time = os.times()[0]

    orig_ppp = part
    min_cell_volume = (min_cell_volume /np.finfo(np.double).eps
        *np.finfo(np.double).eps)

    ispwa = isinstance(ssys, PwaSysDyn)
    islti = isinstance(ssys, LtiSysDyn)

    if ispwa:
        (part, ppp2pwa, part2orig) = pwa_partition(ssys, part)
    else:
        part2orig = range(len(part))

    # Save original polytopes, require them to be convex
    if conservative:
        orig_list = None
        orig = [0]
    else:
        (part, new2old) = part2convex(part) # convexify
        part2orig = [part2orig[i] for i in new2old]

        # map new regions to pwa subsystems
        if ispwa:
            ppp2pwa = [ppp2pwa[i] for i in new2old]

        remove_trans = False # already allowed in nonconservative
        orig_list = []
        for poly in part:
            if len(poly) == 0:
                orig_list.append(poly.copy())
            elif len(poly) == 1:
                orig_list.append(poly[0].copy())
            else:
                raise Exception("discretize: "
                    "problem in convexification")
        orig = list(range(len(orig_list)))

    # Cheby radius of disturbance set
    # (defined within the loop for pwa systems)
    if islti:
        if len(ssys.E) > 0:
            rd = ssys.Wset.chebR
        else:
            rd = 0.

    # Initialize matrix for pairs to check
    IJ = part.adj.copy()
    IJ = IJ.todense()
    IJ = np.array(IJ)
    logger.debug("\n Starting IJ: \n" + str(IJ) )

    # next line omitted in discretize_overlap
    IJ = reachable_within(trans_length, IJ,
                          np.array(part.adj.todense()) )

    # Initialize output
    num_regions = len(part)
    transitions = np.zeros(
        [num_regions, num_regions],
        dtype = int
    )
    sol = deepcopy(part.regions)
    adj = part.adj.copy()
    adj = adj.todense()
    adj = np.array(adj)

    # next 2 lines omitted in discretize_overlap
    if ispwa:
        subsys_list = list(ppp2pwa)
    else:
        subsys_list = None
    ss = ssys

    # init graphics
    if plotit:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.axis('scaled')
            ax2.axis('scaled')
            file_extension = 'pdf'
        except:
            logger.error('failed to import matplotlib')
            plt = None
    else:
    	plt = None

    iter_count = 0

    # List of how many "new" regions
    # have been created for each region
    # and a list of original number of neighbors
    #num_new_reg = np.zeros(len(orig_list))
    #num_orig_neigh = np.sum(adj, axis=1).flatten() - 1

    progress = list()

    # Do the abstraction
    while np.sum(IJ) > 0:
        ind = np.nonzero(IJ)
        # i,j swapped in discretize_overlap
        i = ind[1][0]
        j = ind[0][0]
        IJ[j, i] = 0
        si = sol[i]
        sj = sol[j]

        si_tmp = deepcopy(si)
        sj_tmp = deepcopy(sj)

        #num_new_reg[i] += 1
        #print(num_new_reg)

        if ispwa:
            ss = ssys.list_subsys[subsys_list[i]]
            if len(ss.E) > 0:
                rd, xd = pc.cheby_ball(ss.Wset)
            else:
                rd = 0.

        if conservative:
            # Don't use trans_set
            trans_set = None
        else:
            # Use original cell as trans_set
            trans_set = orig_list[orig[i]]

        S0 = solve_feasible(
            si, sj, ss, N, closed_loop,
            use_all_horizon, trans_set, max_num_poly
        )

        msg = '\n Working with partition cells: ' + str(i) + ', ' + str(j)
        logger.info(msg)

        msg = '\t' + str(i) +' (#polytopes = ' +str(len(si) ) +'), and:\n'
        msg += '\t' + str(j) +' (#polytopes = ' +str(len(sj) ) +')\n'

        if ispwa:
            msg += '\t with active subsystem: '
            msg += str(subsys_list[i]) + '\n'

        msg += '\t Computed reachable set S0 with volume: '
        msg += str(S0.volume) + '\n'

        logger.debug(msg)

        #logger.debug('si \cap s0')
        isect = si.intersect(S0)
        vol1 = isect.volume
        risect, xi = pc.cheby_ball(isect)

        #logger.debug('si \ s0')
        if simu_type == 'bi':
            diff = si.diff(S0)
        elif simu_type == 'dual':
            diff = si
        vol2 = diff.volume
        rdiff, xd = pc.cheby_ball(diff)

        # if pc.is_fulldim(pc.Region([isect]).intersect(diff)):
        #     logging.getLogger('tulip.polytope').setLevel(logging.DEBUG)
        #     diff = pc.mldivide(si, S0, save=True)
        #
        #     ax = S0.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/s0.pdf')
        #
        #     ax = si.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/si.pdf')
        #
        #     ax = isect.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/isect.pdf')
        #
        #     ax = diff.plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/diff.pdf')
        #
        #     ax = isect.intersect(diff).plot()
        #     ax.axis([0.0, 1.0, 0.0, 2.0])
        #     ax.figure.savefig('./img/diff_cap_isect.pdf')
        #
        #     logger.error('Intersection \cap Difference != \emptyset')
        #
        #     assert(False)

        if vol1 <= min_cell_volume:
            logger.warning('\t too small: si \cap Pre(sj), '
                           'so discard intersection')
        if vol1 <= min_cell_volume and isect:
            logger.warning('\t discarded non-empty intersection: '
                           'consider reducing min_cell_volume')
        if vol2 <= min_cell_volume:
            logger.warning('\t too small: si \ Pre(sj), so not reached it')

        # We don't want our partitions to be smaller than the disturbance set
        # Could be a problem since cheby radius is calculated for smallest
        # convex polytope, so if we have a region we might throw away a good
        # cell.
        if (vol1 > min_cell_volume) and (risect > rd) and \
           (vol2 > min_cell_volume) and (rdiff > rd):

            # Make sure new areas are Regions and add proposition lists
            if len(isect) == 0:
                isect = pc.Region([isect], si.props)
            else:
                isect.props = si.props.copy()

            if len(diff) == 0:
                diff = pc.Region([diff], si.props)
            else:
                diff.props = si.props.copy()

            # replace si by intersection (single state)
            isect_list = pc.separate(isect)
            sol[i] = isect_list[0]

            # cut difference into connected pieces
            difflist = pc.separate(diff)

            difflist += isect_list[1:]
            n_isect = len(isect_list) -1

            num_new = len(difflist)

            # add each piece, as a new state
            for region in difflist:
                sol.append(region)

                # keep track of PWA subsystems map to new states
                if ispwa:
                    subsys_list.append(subsys_list[i])
            n_cells = len(sol)
            new_idx = range(n_cells-1, n_cells-num_new-1, -1)

            """Update transition matrix"""
            transitions = np.pad(transitions, (0,num_new), 'constant')

            if(simu_type == 'dual'):
                transitions[new_idx[0], :] = transitions[i, :] # where I stop
                
            transitions[i, :] = np.zeros(n_cells)
            for r in new_idx:
                transitions[:, r] = transitions[:, i] # why comment it? !!!!
                # All sets reachable from start are reachable from both part's
                # except possibly the new part
                transitions[i, r] = 0
                transitions[j, r] = 0

            # sol[j] is reachable from intersection of sol[i] and S0
            if i != j:
                transitions[j, i] = 1

                # sol[j] is reachable from each piece os S0 \cap sol[i]
                #for k in range(n_cells-n_isect-2, n_cells):
                #    transitions[j, k] = 1

            """Update adjacency matrix"""
            old_adj = np.nonzero(adj[i, :])[0]

            # reset new adjacencies
            adj[i, :] = np.zeros([n_cells -num_new])
            adj[:, i] = np.zeros([n_cells -num_new])
            adj[i, i] = 1

            adj = np.pad(adj, (0, num_new), 'constant')

            for r in new_idx:
                adj[i, r] = 1
                adj[r, i] = 1
                adj[r, r] = 1

                if not conservative:
                    orig = np.hstack([orig, orig[i]])

            # adjacencies between pieces of isect and diff
            for r in new_idx:
                for k in new_idx:
                    if r is k:
                        continue

                    if pc.is_adjacent(sol[r], sol[k]):
                        adj[r, k] = 1
                        adj[k, r] = 1

            msg = ''
            if logger.getEffectiveLevel() <= logging.DEBUG:
                msg += '\t\n Adding states ' + str(i) + ' and '
                for r in new_idx:
                    msg += str(r) + ' and '
                msg += '\n'
                logger.debug(msg)

            for k in np.setdiff1d(old_adj, [i,n_cells-1]):
                # Every "old" neighbor must be the neighbor
                # of at least one of the new
                if pc.is_adjacent(sol[i], sol[k]):
                    adj[i, k] = 1
                    adj[k, i] = 1
                elif remove_trans and (trans_length == 1):
                    # Actively remove transitions between non-neighbors
                    transitions[i, k] = 0
                    transitions[k, i] = 0

                for r in new_idx:
                    if pc.is_adjacent(sol[r], sol[k]):
                        adj[r, k] = 1
                        adj[k, r] = 1
                    elif remove_trans and (trans_length == 1):
                        # Actively remove transitions between non-neighbors
                        transitions[r, k] = 0
                        transitions[k, r] = 0

            """Update IJ matrix"""
            IJ = np.pad(IJ, (0,num_new), 'constant')
            adj_k = reachable_within(trans_length, adj, adj)
            sym_adj_change(IJ, adj_k, transitions, i)

            for r in new_idx:
                sym_adj_change(IJ, adj_k, transitions, r)

            if logger.getEffectiveLevel() <= logging.DEBUG:
                msg = '\n\n Updated adj: \n' + str(adj)
                msg += '\n\n Updated trans: \n' + str(transitions)
                msg += '\n\n Updated IJ: \n' + str(IJ)
                logger.debug(msg)

            logger.info('Divided region: ' + str(i) + '\n')
        elif vol2 < abs_tol:
            logger.info('Found: ' + str(i) + ' ---> ' + str(j) + '\n')
            transitions[j,i] = 1
        else:
            if logger.level <= logging.DEBUG:
                msg = '\t Unreachable: ' + str(i) + ' --X--> ' + str(j) + '\n'
                msg += '\t\t diff vol: ' + str(vol2) + '\n'
                msg += '\t\t intersect vol: ' + str(vol1) + '\n'
                logger.debug(msg)
            else:
                logger.info('\t unreachable\n')
            transitions[j,i] = 0

        # check to avoid overlapping Regions
        if debug:
            tmp_part = PropPreservingPartition(
                domain=part.domain,
                regions=sol, adj=sp.lil_matrix(adj),
                prop_regions=part.prop_regions
            )
            assert(tmp_part.is_partition() )

        n_cells = len(sol)
        progress_ratio = 1 - float(np.sum(IJ) ) /n_cells**2
        progress += [progress_ratio]

        msg = '\t total # polytopes: ' + str(n_cells) + '\n'
        msg += '\t progress ratio: ' + str(progress_ratio) + '\n'
        logger.info(msg)

        iter_count += 1

        # no plotting ?
        if not plotit:
            continue
        if plt is None or plot_partition is None:
            continue
        if iter_count % plot_every != 0:
            continue

        tmp_part = PropPreservingPartition(
            domain=part.domain,
            regions=sol, adj=sp.lil_matrix(adj),
            prop_regions=part.prop_regions
        )

        # plot pair under reachability check
        ax2.clear()
        si_tmp.plot(ax=ax2, color='green')
        sj_tmp.plot(ax2, color='red', hatch='o', alpha=0.5)
        plot_transition_arrow(si_tmp, sj_tmp, ax2)

        S0.plot(ax2, color='none', hatch='/', alpha=0.3)
        fig.canvas.draw()

        # plot partition
        ax1.clear()
        plot_partition(tmp_part, transitions.T, ax=ax1, color_seed=23)

        # plot dynamics
        ssys.plot(ax1, show_domain=False)

        # plot hatched continuous propositions
        part.plot_props(ax1)

        fig.canvas.draw()

        # scale view based on domain,
        # not only the current polytopes si, sj
        l,u = part.domain.bounding_box
        ax2.set_xlim(l[0,0], u[0,0])
        ax2.set_ylim(l[1,0], u[1,0])

        if save_img:
            fname = 'movie' +str(iter_count).zfill(3)
            fname += '.' + file_extension
            fig.savefig(fname, dpi=250)
        plt.pause(1)

    new_part = PropPreservingPartition(
        domain=part.domain,
        regions=sol, adj=sp.lil_matrix(adj),
        prop_regions=part.prop_regions
    )

    # check completeness of adjacency matrix
    if debug:
        tmp_part = deepcopy(new_part)
        tmp_part.compute_adj()

    # Generate transition system and add transitions
    ofts = trs.FTS()

    adj = sp.lil_matrix(transitions.T)
    n = adj.shape[0]
    ofts_states = range(n)

    ofts.states.add_from(ofts_states)

    ofts.transitions.add_adj(adj, ofts_states)

    # Decorate TS with state labels
    atomic_propositions = set(part.prop_regions)
    ofts.atomic_propositions.add_from(atomic_propositions)
    for state, region in zip(ofts_states, sol):
        state_prop = region.props.copy()
        ofts.states.add(state, ap=state_prop)

    param = {
        'N':N,
        'trans_length':trans_length,
        'closed_loop':closed_loop,
        'conservative':conservative,
        'use_all_horizon':use_all_horizon,
        'min_cell_volume':min_cell_volume,
        'max_num_poly':max_num_poly
    }

    ppp2orig = [part2orig[x] for x in orig]

    end_time = os.times()[0]
    msg = 'Total abstraction time: ' +\
          str(end_time - start_time) + '[sec]'
    print(msg)
    logger.info(msg)

    if save_img and plt is not None:
        fig, ax = plt.subplots(1, 1)
        plt.plot(progress)
        ax.set_xlabel('iteration')
        ax.set_ylabel('progress ratio')
        ax.figure.savefig('progress.pdf')

    return AbstractPwa(
        ppp=new_part,
        ts=ofts,
        ppp2ts=ofts_states,
        pwa=ssys,
        pwa_ppp=part,
        ppp2pwa=orig,
        ppp2sys=subsys_list,
        orig_ppp=orig_ppp,
        ppp2orig=ppp2orig,
        disc_params=param
    )
