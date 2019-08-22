#!/usr/bin/env python
"""Convert Supremica .wmod format to GraphML .xml format"""

import xml.etree.ElementTree as ET 
import networkx as nx
from tulip.transys import FTS

def wmod_graph_to_FTS(g):
    '''Transform a Networkx graph representing an automaton created from a .wmod file into an FTS'''
    ts = FTS()
    ts.env_actions.add_from(g.graph['uncontrollable'])
    ts.sys_actions.add_from(g.graph['controllable'])
    print(ts.env_actions)
    print(ts.sys_actions)
    ts.atomic_propositions.add_from(g.graph['props'])
    for node, node_data in g.nodes(data=True):
        node_ap = g.graph['props'] & node_data.keys()
        ts.add_node(node, ap=node_ap)

    ts.states.initial.add_from(g.graph['init_nodes'])

    for source, dest, data in g.edges(data=True):
        if data['event'] in g.graph['uncontrollable']:
            ts.transitions.add(source, dest, env_actions=data['event'])
        else:
            ts.transitions.add(source, dest, sys_actions=data['event'])
    return ts

def read_wmod_as_graph(inputpath):
    '''Read in a .wmod file exported from Supremica as a Networkx graph
    '''
    
    wmod = ET.parse(inputpath) 
    root = wmod.getroot()
    ns = {'d': "http://waters.sourceforge.net/xsd/module"}
    
    g = nx.MultiDiGraph()
    g.graph['name'] = root.get('Name')
    
    controllable = set()
    uncontrollable = set()
    props = set()
    
    event_list = root.find('d:EventDeclList', ns)
    for event_dec in event_list.findall('d:EventDecl', ns):
        event_kind = event_dec.get('Kind')
        event_name = event_dec.get('Name')
        if event_kind == "PROPOSITION":
            props.add(event_name)
        elif event_kind == "CONTROLLABLE":
            controllable.add(event_name)
        elif event_kind == "UNCONTROLLABLE":
            uncontrollable.add(event_name)
    
    g.graph['props'] = props
    g.graph['controllable'] = controllable
    g.graph['uncontrollable'] = uncontrollable

    comp_list = root.find('d:ComponentList',ns)
    simple_comp = comp_list.find('d:SimpleComponent', ns)
    graph = simple_comp.find('d:Graph', ns)
    node_list = graph.find('d:NodeList', ns)
    init_nodes = []
    for node in node_list.findall('d:SimpleNode', ns):
        node_name = node.get("Name")
        g.add_node(node_name)
        if node.get("Initial") == "true":
            g.node[node_name]['initial'] = True
            init_nodes.append(node_name)
        node_event_list = node.find('d:EventList', ns)
        if not (node_event_list is None):
            for ident in node_event_list.findall('d:SimpleIdentifier', ns):
                ident_name = ident.get('Name')
                g.node[node_name][ident_name] = True
    
    g.graph['init_nodes'] = init_nodes
    
    edge_list =  graph.find('d:EdgeList', ns)
    for edge in edge_list.findall('d:Edge', ns):
        source = edge.get('Source')
        target = edge.get('Target')
        event_ident = edge.find('d:LabelBlock', ns).find('d:SimpleIdentifier', ns)
        event_name = event_ident.get('Name')
        g.add_edge(source, target, event=event_name)
    return g
    
    
def read_xml_automaton_as_graph(inputpath):
    '''Read an automaton represented by XML into a Networkx graph
    This function is still being developed.
    Such an XML file is produced by exporting an automaton from Supremica in the analyze tab
    '''
    
    wmod = ET.parse(inputpath) 
    root = wmod.getroot()
    
    g = nx.MultiDiGraph()
    
    automaton = root.find('Automaton')
    g.graph['name']=automaton.get("name")
    
    event_dict = {}
    state_dict = {}
    
    events = automaton.find('Events')
    for event in events.findall('Event'):
        id = event.get('id')
        label = event.get('label')
        event_dict[id] = label
    
    states = automaton.find('States')
    for state in states.findall('State'):
        id = state.get('id')
        name = state.get('name')
        state_dict[id] = name
        g.add_node(name)
        if state.get('initial') == "true":
            g.node[name]['initial'] = True
    
    print(state_dict)
    transitions = automaton.find('Transitions')
    for trans in transitions.findall('Transition'):
        source = trans.get('source')
        dest = trans.get('dest')
        event = trans.get('event')
        g.add_edge(state_dict[source], state_dict[dest], event=event_dict[event])
    
    return g