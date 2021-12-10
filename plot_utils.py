import networkx as nx
import scipy.stats as ss
import plotly.graph_objects as go

#sample plot taken from https://plotly.com/python/network-graphs/

def plot_network(*, graph, tics=None, layout='planar', title="", min_node_degree_to_show_label=10, top_perc_to_show_label=0.05):
    
    if tics is not None:
        mapping = dict([(index, tic) for index, tic in enumerate(tics)])
        graph = nx.relabel.relabel_nodes(graph, mapping, copy=True)

    if layout == 'planar':
        nodePos = nx.planar_layout(graph)
    elif layout == 'circular':
        G = nx.Graph()
        G.add_nodes_from(list(range(len(graph.nodes))) if tics is None else tics)
        nodePos = nx.circular_layout(G)
    else:
        nodePos = nx.planar_layout(graph)

    for node in graph.nodes:
        graph.nodes[node]['pos'] = nodePos[node]

    edge_x = []
    edge_y = []
    weights = []
    for edge in graph.edges():
        x0, y0 = graph.nodes[edge[0]]['pos']
        x1, y1 = graph.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        weights.append(f"weight: {graph.get_edge_data(edge[0], edge[1])['weight']}")
        
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        #hoverinfo='none',
        hoverinfo='text',
        mode='lines')
    edge_trace.text = weights

    node_x = []
    node_y = []
    for node in graph.nodes():
        x, y = graph.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node degree',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for adjacencies in graph.adjacency():
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]}# of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=title,
                    titlefont_size=20,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    
    node_degrees_ranks = ss.rankdata([-len(adjacencies[1]) for adjacencies in graph.adjacency()], method='min')
    top_n_to_show_label = top_perc_to_show_label * len(node_degrees_ranks)
    for index, adjacencies in enumerate(graph.adjacency()):
        if len(adjacencies[1]) > min_node_degree_to_show_label and node_degrees_ranks[index] <= top_n_to_show_label:
            fig.add_annotation(x=graph.nodes[adjacencies[0]]['pos'][0], 
                               y=graph.nodes[adjacencies[0]]['pos'][1],
                               text=f"{adjacencies[0]} #{len(adjacencies[1])}",
                               yshift=5 if layout in ['planar'] else 15,
                               showarrow=True if layout in ['planar'] else False,
                               arrowhead=1)
    fig.show()