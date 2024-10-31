# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import ipywidgets as widgets
from IPython.display import display
from Graph import Graph

# Load the combined JSON file with all snapshots
with open('/home/lichensong/projects/AlphaZeroArcade/goldenfiles/mcts_tests/tictactoe_uniform.json', 'r') as f:
    graph_data = json.load(f)

# Extract the list of snapshots from the JSON
graph_snapshots = graph_data['graphs']

graph = Graph(graph_snapshots)

# Create an interactive slider to select the snapshot index
snapshot_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=len(graph_snapshots) - 1,
    step=1,
    description='Snapshot:',
    continuous_update=False
)

# Display the plot for the selected snapshot whenever the slider value changes
widgets.interactive(lambda index: graph.plot_snapshot(index), index=snapshot_slider)
# -


