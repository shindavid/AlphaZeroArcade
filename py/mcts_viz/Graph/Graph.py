import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, graph_snapshots):
        self.graph_snapshots = graph_snapshots

    def plot_snapshot(self, index):
        G = self.create_graph(index)
        G_prev = self.create_graph(index - 1) if index > 0 else None
        new_nodes, changed_nodes, new_edges, changed_edges = self.get_diff(G_prev, G)

        # Use graphviz_layout for hierarchical placement
        pos = graphviz_layout(G, prog="dot")

        # Define node labels, coloring new or changed nodes in red
        node_labels = {}
        for node, data in G.nodes(data=True):
            label = f"({node})\n{data['state']}\nN: {data['N']}\nQ: [{float(data['Q'][0]):.2f}, {float(data['Q'][1]):.2f}]"
            color = "red" if node in new_nodes or node in changed_nodes else "black"
            node_labels[node] = (label, color)

        # Define edge labels, coloring new or changed edges in red
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            label = f"{data['index']}\nA: {data['action']}, E: {data['E']}"
            color = "red" if (u, v) in new_edges or (u, v) in changed_edges else "black"
            edge_labels[(u, v)] = (label, color)

        # Clear the current plot
        plt.figure(figsize=(20, 10))

        # Draw nodes and labels, applying color to changed/new nodes
        nx.draw(
            G, pos, with_labels=False, node_size=3000, node_color="lightblue", font_size=10
        )
        for node, (label, color) in node_labels.items():
            nx.draw_networkx_labels(G, pos, labels={node: label}, font_size=10, font_color=color)

        # Draw edges and edge labels, applying color to new or changed edges
        for (u, v), (label, color) in edge_labels.items():
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=color)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: v[0] for k, v in edge_labels.items()}, label_pos=0.6, font_size=8)

        # Set the plot title to indicate which snapshot is being displayed
        plt.title(f"DAG Snapshot {index + 1}")
        plt.show()

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

if __name__ == '__main__':
    import json
    with open('./py/alphazero/dashboard/Graph/graph_jsons/tictactoe_uniform.json', 'r') as f:
        graph_data = json.load(f)

    graph_snapshots = graph_data['graphs']
    graph = Graph(graph_snapshots)
    graph.plot_snapshot(9)