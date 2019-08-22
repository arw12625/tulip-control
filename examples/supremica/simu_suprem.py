#!/usr/bin/env python
"""Test working with files from Supremica"""
import logging

import numpy as np
from polytope import box2poly
from tulip import hybrid
from tulip.abstract import discretize
from tulip.abstract import prop2part
from tulip.transys.transys import FTS
from tulip.transys.transys import simu_abstract
from examples.supremica import supremica_io

logging.basicConfig()
logger = logging.getLogger(__name__)


def build_FTS(index):
    # build test FTS
    # simple test
    ts = None
    if index == 0:
        wmod_file_path = 'data/egen.wmod'
        g = supremica_io.read_wmod_as_graph(wmod_file_path)
        ts = supremica_io.wmod_graph_to_FTS(g)

    return ts


def simu_abstract_test():
    for index in range(1):
        ts = build_FTS(index)
        bi_simu, bi_part = simu_abstract(ts, 'bi')
        #dual_simu, dual_part = simu_abstract(ts, 'dual')
        # check dual-simulation of bi_simu
        # pick the smallest cell in dual_simu for each state in ts
        '''K12 = dual_part['ts2simu'].copy()
        for i in K12:
            list_s2 = K12[i]
            point = 0
            curr_len = 0
            best_len = 1e10
            for j in list_s2:
                curr_len = len(dual_part['simu2ts'][j])
                if curr_len < best_len:
                    best_len = curr_len
                    point = j
            K12[i] = set([point])
        assert check_simulation(ts, dual_simu, K12,
                                dual_part['simu2ts'])
        assert check_simulation(dual_simu, ts, dual_part['simu2ts'],
                                dual_part['ts2simu'])
        # check bisimulation of bi_simu
        assert check_simulation(ts, bi_simu, bi_part['ts2simu'],
                                bi_part['simu2ts'])
        assert check_simulation(bi_simu, ts, bi_part['simu2ts'],
                                bi_part['ts2simu'])
        '''
        print(bi_simu)

if __name__ == '__main__':
    simu_abstract_test()
