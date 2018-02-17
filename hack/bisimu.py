#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 15:07:56 2018

@author: zexiang
"""
from tulip.transys.transys import FTS, tuple2fts
import networkx as nx
import Queue

def pre(graph,list_n):
        """Predecessor
        
        @param graph: graph defined in C{MultiDiGraph}
        @param list_n: list of nodes
        @return: set of predecessors of C{list_n}
        """
        #The simplest pre is implemented. Modify this fun to ContPre if necessary.
        pre_set = set()
        for n in list_n:
            pre_set = pre_set.union(graph.predecessors(n))
        return pre_set

def bisimu(ts):
    """Create a bisimulation abstraction for a Finite Transition System.
    
    @param ts: L{FTS}
    @return: L{FTS}
    
    References
    ==========
    1. Wagenmaker, A. J.; Ozay, N.
    A Bisimulation-like Algorithm for Abstracting Control Systems. 
    54th Annual Allerton Conference on CCC 2016, 569–576.
    """
    # recover the FTS to MultiDiGraph
    G = nx.MultiDiGraph(ts)
    
    # build coarsest partition (graph + hash table)
    S0 = dict()
    G0 = nx.MultiDiGraph() # a graph associated with the coarsest partition
    num_cell = 0
    
    for i in G:
        node = G.node.keys()[i]
        ap = str(G.node[node]['ap'])
        if not S0.has_key(ap):
            S0[ap]=set()
            G0.add_node(num_cell,ap=ap,cov=S0[ap]) # hash table S0--->G
            num_cell += 1
            
        S0[ap].add(node)
    
    # build a queue of node, used for the while loop
    queue = Queue.Queue()
    # add edges in the coarsest partition, and add nodes in the queue
    for i in G0:
        pre_i = pre(G,G0.node[i]['cov'])    
        for j in G0:
            cov_j = G0.node[j]['cov']
            if pre_i.intersection(cov_j)!=set():
                G0.add_edge(j,i)
                
        queue.put(i)
           
    # bisimulation while loop
    while not queue.empty():# queue is empty
        # pop a node from the queue
        i = queue.get()
        # calculuate its pre in G
        pre_i = pre(G,G0.node[i]['cov'])
        # intersect the pre of node with states of other nodes
        for j in G0.predecessors(i):
            cov_j = G0.node[j]['cov']
            
            if pre_i.intersection(cov_j)==set():
                G0.remove_edge(j,i)
                continue
            
            if not cov_j.issubset(pre_i):
                #  add new node and update the graph 
                G0.node[j]['cov'] = cov_j.intersection(pre_i)
                G0.add_node(num_cell,ap=G0.node[j]['ap'],cov=cov_j.difference(pre_i))
                G0.add_edges_from([(j,num_cell),(num_cell,j)])                
                [G0.add_edge(k,num_cell) for k in G0.predecessors(j)]
                [G0.add_edge(num_cell,k) for k in G0.successors(j)]
                
                # add effected cells into the queue
                queue.put(j)
                queue.put(num_cell)
                num_cell += 1
                if(i==j):
                    break
#        print 'the queue size is', queue.qsize()
#        print 'num of cell is', num_cell
    return G0

def dual_simu(ts):
    """Create a dual-simulation abstraction for a Finite Transition System.
    
    @param ts: L{FTS}
    @return: L{FTS}
    
    References
    ==========
    1. Wagenmaker, A. J.; Ozay, N.
    A Bisimulation-like Algorithm for Abstracting Control Systems. 
    54th Annual Allerton Conference on CCC 2016, 569–576.
    """
    # recover the FTS to MultiDiGraph
    G = nx.MultiDiGraph(ts)
    
    # build coarsest partition (graph + hash table)
    S0 = dict()
    G0 = nx.MultiDiGraph() # a graph associated with the coarsest partition
    num_cell = 0
    
    for i in G:
        node = G.node.keys()[i]
        ap = str(G.node[node]['ap'])
        if not S0.has_key(ap):
            S0[ap]=set()
            G0.add_node(num_cell,ap=ap,cov=S0[ap]) # hash table S0--->G
            num_cell += 1
            
        S0[ap].add(node)
    
    # build a queue of node, used for the while loop
    queue = Queue.Queue()
    # add edges in the coarsest partition, and add nodes in the queue
    for i in G0:
        pre_i = pre(G,G0.node[i]['cov'])    
        for j in G0:
            cov_j = G0.node[j]['cov']
            if pre_i.intersection(cov_j)!=set():
                G0.add_edge(j,i)
                
        queue.put(i)
           
    # bisimulation while loop
    while not queue.empty():# queue is empty
        # pop a node from the queue
        i = queue.get()
        # calculuate its pre in G
        pre_i = pre(G,G0.node[i]['cov'])
        # intersect the pre of node with states of other nodes
        for j in G0.predecessors(i):
            cov_j = G0.node[j]['cov']
            
            if pre_i.intersection(cov_j)==set():
                G0.remove_edge(j,i)
                continue
            
            if not cov_j.issubset(pre_i):
                #  add new node and update the graph 
                G0.node[j]['cov'] = cov_j.intersection(pre_i)
                G0.add_node(num_cell,ap=G0.node[j]['ap'],cov=cov_j)
                G0.add_edges_from([(j,num_cell),(num_cell,j)])                
                [G0.add_edge(k,num_cell) for k in G0.predecessors(j)]
                [G0.add_edge(num_cell,k) for k in G0.successors(j)]
                G0.remove_edge(num_cell,i) # remove wrong edge in coarser part
                # add effected cells into the queue
                queue.put(j)
                queue.put(num_cell)
                num_cell += 1
                if(i==j):
                    break
#        print 'the queue size is', queue.qsize()
#        print 'num of cell is', num_cell
    return G0


def simu_abstract(ts,simu_type):
    """Create a bi/dual-simulation abstraction for a Finite Transition System.
    
    @param ts: L{FTS}
    @param simu_type: string 'bi'/'dual', flag used to switch b.w. bisimu and dual-simu
    @return: L{FTS}
    
    References
    ==========
    1. Wagenmaker, A. J.; Ozay, N.
    A Bisimulation-like Algorithm for Abstracting Control Systems. 
    54th Annual Allerton Conference on CCC 2016, 569–576.
    """
    def pre(graph,list_n):
        """Predecessor
        
        @param graph: graph defined in C{MultiDiGraph}
        @param list_n: list of nodes
        @return: set of predecessors of C{list_n}
        """
        #The simplest pre is implemented. Modify this fun to ContPre if necessary.
        pre_set = set()
        for n in list_n:
            pre_set = pre_set.union(graph.predecessors(n))
        return pre_set
    
    # recover the FTS to MultiDiGraph
    G = nx.MultiDiGraph(ts)
    
    # build coarsest partition (graph + hash table)
    S0 = dict()
    G0 = nx.MultiDiGraph() # a graph associated with the coarsest partition
    num_cell = 0
    
    for node in G:
        ap = repr(G.node[node]['ap'])
        if not S0.has_key(ap):
            S0[ap]=set()
            G0.add_node(num_cell,ap=ap,cov=S0[ap]) # hash table S0--->G
            num_cell += 1
            
        S0[ap].add(node)
    
    # build a queue of node, used for the while loop
    queue = Queue.Queue()
    # add edges in the coarsest partition, and add nodes in the queue
    for i in G0:
        pre_i = pre(G,G0.node[i]['cov'])    
        for j in G0:
            cov_j = G0.node[j]['cov']
            if pre_i.intersection(cov_j)!=set():
                G0.add_edge(j,i)
                
        queue.put(i)
           
    # bisimulation while loop
    while not queue.empty():# queue is empty
        # pop a node from the queue
        i = queue.get()
        # calculuate its pre in G
        pre_i = pre(G,G0.node[i]['cov'])
        # intersect the pre of node with states of other nodes
        for j in G0.predecessors(i):
            cov_j = G0.node[j]['cov']
            
            if pre_i.intersection(cov_j)==set():
                G0.remove_edge(j,i)
                continue
            
            if not cov_j.issubset(pre_i):
                #  add new node and update the graph 
                G0.node[j]['cov'] = cov_j.intersection(pre_i)
                if(simu_type=='dual'):
                    G0.add_node(num_cell,ap=G0.node[j]['ap'],cov=cov_j)
                elif (simu_type=='bi'):
                    G0.add_node(num_cell,ap=G0.node[j]['ap'],cov=cov_j.difference(pre_i))
                                  
                [G0.add_edge(k,num_cell) for k in G0.predecessors_iter(j)]
                [G0.add_edge(num_cell,k) for k in G0.successors_iter(j)]
#                G0.add_edges_from([(j,num_cell),(num_cell,j)])  
                G0.remove_edge(num_cell,i) # remove wrong edge in coarser part
                # add effected cells into the queue
                queue.put(j)
                queue.put(num_cell)
                num_cell += 1
                if(i==j):
                    break
#        print 'the queue size is', queue.qsize()
#        print 'num of cell is', num_cell
                    
    # construct new FTS
    env_actions = [
                {'name': 'env_actions',
                 'values': ts.env_actions,
                 'setter': True}]
    sys_actions = [
                {'name': 'sys_actions',
                 'values': ts.sys_actions,
                 'setter': True}]
    ts_simu = FTS(env_actions,sys_actions)
    ts2simu = {}
    for i in G0:
        for j in G0.node[i]['cov']:
            if(ts2simu.has_key(j)):
                ts2simu[j].append(i)
            else:
                ts2simu[j]=[i]
            
    S = G0.nodes()
    S0 = set()
    for i in ts.states.initial:
        [S0.add(j) for j in ts2simu[i]]
    AP = ts.aps
    
    L = []
    for i in G0:
        L.append((i,eval(G0.node[i]['ap'])))
    # do we consider actions for bi/dual simulation abstraction?
#    if((len(ts.actions['sys_actions'])+len(ts.actions['env_actions']))==0):
    Act = None
#    else:
#        Act = ts.actions
    
    trans=[]
#    if((len(ts.actions['sys_actions'])+len(ts.actions['env_actions']))!=0):
#        for i,j in G0.edges_iter():
#            trans.append((i,j,G0.edge[i][j]))
#    else:
    for i,j in G0.edges_iter():
        trans.append((i,j))    
        
    ts_simu = tuple2fts(S,S0,AP,L,Act,trans,name=simu_type,prepend_str='')
    
#    ts_simu.add_nodes_from(G0.nodes())
#    ts_simu.states.initial.add(S0)
#    ts_simu.add_edges_from(G0.edges())
#    ts_simu.atomic_propositions.add_from(ts.atomic_propositions)
#    for i in ts_simu:
#        ts_simu.states.add(i, ap=G0.node[i]['ap'])
    # build hash-table mapping nodes in ts to nodes in G0
    
    return ts_simu, G0
