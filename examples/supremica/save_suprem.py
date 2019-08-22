#!/usr/bin/env python

import networkx as nx
from examples.supremica import supremica_io
import pickle

'''
xml_file_path = 't.xml'
g = wmod_converter.read_xml_automaton_as_graph(xml_file_path)
print(g.node)
print(g.edge)
'''

wmod_file_path = 'data/egen.wmod'

g = supremica_io.read_wmod_as_graph(wmod_file_path)
print(g.node)
print(g.edge)
print(g.graph)

pickle.dump( g, open( "data/egen.p", "wb" ) )