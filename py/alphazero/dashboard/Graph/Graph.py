import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

class Graph:
  def __init__(self, graph_snapshots):
      self.graph_snapshots = graph_snapshots

  def plot_snapshot(self, index):
      G = nx.DiGraph()
      snapshot = self.graph_snapshots[index]
      prev_snapshot = self.graph_snapshots[index - 1] if index > 0 else None
      new_nodes, changed_nodes = self.add_nodes(G, snapshot, prev_snapshot)
      new_edges, changed_edges = self.add_edges(G, snapshot, prev_snapshot)

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

  def add_nodes(self, G, snapshot, prev_snapshot):
      new_nodes = set()
      changed_nodes = set()

      # Add nodes, tracking changes
      for node in snapshot['nodes']:
          node_id = node['index']
          G.add_node(node_id, N=node['N'], Q=node['Q'], state=node['state'])

          # Check if the node is new or has changed if a previous snapshot exists
          if prev_snapshot:
              prev_node = next((n for n in prev_snapshot['nodes'] if n['index'] == node_id), None)
              if not prev_node:
                  new_nodes.add(node_id)  # New node
              elif node['N'] != prev_node['N'] or node['Q'] != prev_node['Q']:
                  changed_nodes.add(node_id)  # Changed node
          else:
              new_nodes.add(node_id)  # No previous snapshot, mark as new

      return (new_nodes, changed_nodes)

  def add_edges(self, G, snapshot, prev_snapshot):
      new_edges = set()
      changed_edges = set()

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

          if prev_snapshot:
              prev_edge = next((e for e in prev_snapshot['edges'] if e['from'] == edge['from'] and e['to'] == edge['to']), None)
              if not prev_edge:
                  new_edges.add((from_node, to_node))  # New edge
              elif edge['E'] != prev_edge['E'] or edge['action'] != prev_edge['action']:
#                     print(f"{edge['index']} edge E: {edge['E']}, prev_edge E: {prev_edge['E']}; edge A: {edge['action']}, prev_edge A: {prev_edge['action']}")
                  changed_edges.add((from_node, to_node))  # Edge label has changed
          else:
              new_edges.add((from_node, to_node))  # No previous snapshot, mark as new

      return (new_edges, changed_edges)
