import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import json

class Graph:
    def __init__(self, graph_snapshots):
        self.graph_snapshots = graph_snapshots

    def create_graph(self, index):
        G = nx.DiGraph()
        snapshot = self.graph_snapshots[index]
        self.insert_nodes(G, snapshot)
        self.insert_edges(G, snapshot)
        return G

    def insert_nodes(self, G, snapshot):
        for node in snapshot['nodes']:
            node_id = node['index']
            G.add_node(node_id, N=node['N'], Q=node['Q'], state=node['state'])

    def insert_edges(self, G, snapshot):
        # Add edges, tracking changes
        for edge in snapshot['edges']:
            from_node = edge['from']
            to_node = edge['to']

            edge_data = G.get_edge_data(from_node, to_node)

            if G.has_edge(from_node, to_node):
                G[from_node][to_node]['index'].append(edge['index'])
                G[from_node][to_node]['E'] += edge['E']
                G[from_node][to_node]['action'].append(edge['action'])
            else:
                G.add_edge(from_node, to_node, E=edge['E'], action=[edge['action']], index=[edge['index']])

    def get_diff(self, G0, G1):
        if G0 is None:
            return (set(G1.nodes), set(), set(G1.edges), set())
        # Get the nodes and edges that are new or changed
        new_nodes = set(G1.nodes) - set(G0.nodes)
        changed_nodes = {node for node in G0.nodes if G0.nodes[node]['N'] != G1.nodes[node]['N'] or G0.nodes[node]['Q'] != G1.nodes[node]['Q']}
        new_edges = set(G1.edges) - set(G0.edges)
        changed_edges = {edge for edge in G0.edges if G0.edges[edge]['E'] != G1.edges[edge]['E'] or G0.edges[edge]['action'] != G1.edges[edge]['action']}

        return (new_nodes, changed_nodes, new_edges, changed_edges)

