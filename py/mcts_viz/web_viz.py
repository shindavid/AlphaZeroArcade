from Graph import Graph

import json
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import os
import numpy as np

app = Dash(__name__, suppress_callback_exceptions=True)

# Load your JSON data
base_dir = '/home/lichensong/projects/AlphaZeroArcade/sample_search_logs/mcts_tests'

# List all files ending with '_log.json'
log_files = [f for f in os.listdir(base_dir) if f.endswith('_log.json')]

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': file, 'value': file} for file in log_files],
        placeholder='Select a file',
        style={'width': '50%'}
        ),
    dcc.Store(id='graph-data-store'),
    html.Div(id='dynamic-layout')
    ])

@app.callback(
    [Output('dynamic-layout', 'children'),
     Output('graph-data-store', 'data')],
    Input('file-dropdown', 'value')
)
def update_layout(selected_file):
    if selected_file is None:
        return html.Div(), None  # Return empty until a file is selected

    json_file_path = os.path.join(base_dir, selected_file)
    with open(json_file_path, 'r') as f:
        graph_data = json.load(f)
    graph_snapshots = graph_data['graphs']

    # Once a file is selected, add the slider and graph
    return html.Div([
        dcc.Slider(
            id='snapshot-slider',
            min=0,
            max=len(graph_snapshots) - 1,
            value=0,
            step=1
        ),
        dcc.Graph(id='network-graph')
    ]), graph_snapshots

# Define callback to update graph
@app.callback(
    Output('network-graph', 'figure'),
    [Input('snapshot-slider', 'value'),
     Input('graph-data-store', 'data')])
def update_graph(index, graph_snapshots):
    graph = Graph(graph_snapshots)
    G = graph.create_graph(index)
    G_prev = graph.create_graph(index - 1) if index > 0 else None
    new_nodes, changed_nodes, new_edges, changed_edges = graph.get_diff(G_prev, G)

    pos = graphviz_layout(G, prog="dot")
    traces = []

    # Create one trace per edge with appropriate color
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xm = x0 * 0.8 + x1 * 0.2
        ym = y0 * 0.8 + y1 * 0.2

        if (u, v) in new_edges:
            edge_color = 'red'
        elif (u, v) in changed_edges:
            edge_color = 'blue'
        else:
            edge_color = 'gray'

        edge_label = f"{data['index']}<br>A: {data['action']}, E: {data['E']}"

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1.0, color=edge_color),
            mode='lines')
        traces.append(edge_trace)

        # Create a separate trace for the edge label
        label_trace = go.Scatter(
            x=[xm],
            y=[ym],
            text=[edge_label],
            mode='text',
            textposition='middle center',
            hoverinfo='none'
        )
        traces.append(label_trace)

    # Create trace for nodes
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    for node, data in G.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if node in changed_nodes:
            node_colors.append('lightblue')
        elif node in new_nodes:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightgray')

        label = f"({node})<br>{data['state']}<br>N: {data['N']}<br>Q: [{float(data['Q'][0]):.2f}, {float(data['Q'][1]):.2f}]"
        node_text.append(label)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            colorscale='YlGnBu',
            size=50,
            color=node_colors,
            line=dict(color='lightgrey', width=0.5)))

    traces.append(node_trace)

    figure = go.Figure(data=traces,
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           width=1600,
                           height=1000,
                           margin=dict(b=0, l=0, r=0, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )

    return figure

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)