import networkx as nx
import scipy.stats as ss
import plotly.graph_objects as go
import numpy as np

#sample plot taken from https://plotly.com/python/network-graphs/
def plot_network(*, graph, layout='planar', title="", min_node_degree_to_show_label=10, top_perc_to_show_label=0.05):

    if layout == 'planar':
        nodePos = nx.planar_layout(graph)
    elif layout == 'circular':
        G = nx.Graph()
        G.add_nodes_from(list(range(len(graph.nodes))))
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
            size=[],
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
        node_text.append(f'{graph.nodes[adjacencies[0]]["ticker"]}# of connections: {len(adjacencies[1])}')
    
    node_trace.marker.size = [6 + 7 * np.log(node_adjacency / 3) for node_adjacency in node_adjacencies]
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
                               text=f'{graph.nodes[adjacencies[0]]["ticker"]} #{len(adjacencies[1])}',
                               yshift=5 if layout in ['planar'] else 15,
                               showarrow=True if layout in ['planar'] else False,
                               arrowhead=1)
    fig.show()

def plot_networks(*, graphs_1, graphs_2=None, label_graph_1='PMFG', label_graph_2='MST',  layout='planar', title="", min_node_degree_to_show_label=10, top_perc_to_show_label=0.15, show_nodes_with_nan=True):
    graphs_1 = graphs_1.copy()
    if graphs_2:
        graphs_2 = graphs_2.copy()
    
    times = sorted(graphs_1.keys())

    if show_nodes_with_nan:
        def get_set_all_tickers(graphs):
            return set(x for l in [nx.get_node_attributes(graph, "ticker").values() for _, graph in graphs.items()] for x in l)
        ticker_all_graphs = list(get_set_all_tickers(graphs_1).union(get_set_all_tickers(graphs_2))) if graphs_2 else list(get_set_all_tickers(graphs_1))


        def update_graphs(graphs, ticker_all_graphs):
            for time, graph in graphs.items():
                ticker = nx.get_node_attributes(graph, "ticker")
                mapping = dict()
                for index, tick in ticker.items():
                    mapping[index] = ticker_all_graphs.index(tick)

                graph_new_labels = nx.relabel_nodes(graph, mapping)
                graph_new_labels.add_nodes_from(set(list(range(len(ticker_all_graphs)))).difference(set(list(graph_new_labels.nodes))))
                
                nx.set_node_attributes(graph_new_labels, dict([(tic_index, tic) for tic_index, tic in enumerate(ticker_all_graphs)]), 'ticker')
                graphs[time] = graph_new_labels
        
        update_graphs(graphs_1, ticker_all_graphs)
        if graphs_2:
            update_graphs(graphs_2, ticker_all_graphs)
        
    max_node_degree = max([len(adjacencies[1]) for _, graph in graphs_1.items() for adjacencies in graph.adjacency()])
    if graphs_2:
        max_node_degree_graphs_2 = max([len(adjacencies[1]) for _, graph in graphs_2.items() for adjacencies in graph.adjacency()])
        max_node_degree = max(max_node_degree, max_node_degree_graphs_2)

    # contains 2 traces (lines and dots) for each graph 
    traces = []
    #contains annotations for each graph
    annotations_list = []
    titles = []
    slider_ticks = []
        
    for time in times:
        graph_1 = graphs_1[time]
        graph_2 = graphs_2[time] if graphs_2 else None
        if layout == 'planar':
            nodePos = nx.planar_layout(graph_1)
        elif layout == 'circular':
            G = nx.Graph()
            G.add_nodes_from(list(range(len(graph_1.nodes))))
            nodePos = nx.circular_layout(G)
        else:
            nodePos = nx.planar_layout(graph_1)

        nx.set_node_attributes(graph_1, nodePos , 'pos')
        if graphs_2:
            nx.set_node_attributes(graph_2, nodePos , 'pos')
        
        node_x = []
        node_y = []
        for node in graph_1.nodes():
            x, y = graph_1.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=False,
                color=[],
                size=[],
                colorbar=dict(
                    thickness=15,
                    title='Node degree',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2),
            visible=False,
            name=f"Color: #{label_graph_1}<br>Size: #{label_graph_2}",
            showlegend=True if graphs_2 else False)

        edge_x = []
        edge_y = []
        weights =[]
        weights_text = []
        for edge in graph_1.edges():
            x0, y0 = graph_1.nodes[edge[0]]['pos']
            x1, y1 = graph_1.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
            weights.append(graph_1.get_edge_data(edge[0], edge[1])['weight'])
            weights_text.append(f"weight: {graph_1.get_edge_data(edge[0], edge[1])['weight']}")
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, 
                      color='#888'),
            #hoverinfo='none',
            hoverinfo='text',
            mode='lines',
            visible=False,
            name=label_graph_1)
        edge_trace.text = weights

        if graphs_2:
            edge_x = []
            edge_y = []
            weights =[]
            weights_text = []
            for edge in graph_2.edges():
                x0, y0 = graph_2.nodes[edge[0]]['pos']
                x1, y1 = graph_2.nodes[edge[1]]['pos']
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)
                weights.append(graph_2.get_edge_data(edge[0], edge[1])['weight'])
                weights_text.append(f"weight: {graph_2.get_edge_data(edge[0], edge[1])['weight']}")
                
            edge_trace_mst = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, 
                        color='#ff3221'),
                #hoverinfo='none',
                hoverinfo='text',
                mode='lines',
                visible=False,
                name=label_graph_2)
            edge_trace_mst.text = weights

        node_adjacencies_graph_1 = []
        node_adjacencies_graph_2 = []
        node_text = []
        if graph_2:
            graph_2_adjacency_dict = dict(graph_2.adjacency())
        for node, adjecent_nodes in graph_1.adjacency():
            node_adjacencies_graph_1.append(len(adjecent_nodes))
            if graphs_2:
                node_adjacencies_graph_2.append(len(graph_2_adjacency_dict[node]))
            else:
                node_text.append(f"{graph_1.nodes[node]['ticker']} # of connections: {len(adjecent_nodes)}")
        #to keep scale same for each time we add max to all the subplots
        node_adjacencies_graph_1.append(max_node_degree)
        
        node_trace.marker.color = node_adjacencies_graph_1
        if graph_2 is None:
            node_adjacencies_graph_2 = node_adjacencies_graph_1
        
        node_trace.marker.size = [6 + 7 * np.log(node_adjacency / 3 + 1) for node_adjacency in node_adjacencies_graph_2]

        #node_trace.text = node_text

        #traces.append(edge_trace)
        #traces.append(node_trace)
        #traces.append(edge_trace_mst)

        def get_annotations(graph):
            annotations = []
            node_degrees_ranks = ss.rankdata([-len(adjacent_nodes) for node, adjacent_nodes in graph.adjacency()], method='min')
            top_n_to_show_label = top_perc_to_show_label * len(node_degrees_ranks)

            for index, (node, adjacent_nodes) in enumerate(graph.adjacency()):
                if len(adjacent_nodes) > min_node_degree_to_show_label and node_degrees_ranks[index] <= top_n_to_show_label:
                    annotations.append(dict(x=graph.nodes[node]['pos'][0], 
                                            y=graph.nodes[node]['pos'][1],
                                            text=f"{graph.nodes[node]['ticker']} #{len(adjacent_nodes)}",
                                            #text=f"{node} #{len(adjacent_nodes)}",
                                            yshift=5 if layout in ['planar'] else 15,
                                            showarrow=True if layout in ['planar'] else False,
                                            arrowhead=1
                                            )
                                        )
            return annotations

#        annotations_list.append([annotation for _, _, annotation in get_annotations(graph_1)])
        def get_annotation_and_node_text(graph_1, graph_2):
            def get_annotations_data(graph):
                annotations_data = []
                node_degrees_ranks = ss.rankdata([-len(adjacent_nodes) for node, adjacent_nodes in graph.adjacency()], method='min')
                for index, (node, adjacent_nodes) in enumerate(graph.adjacency()):
                    rank = node_degrees_ranks[index]
                    annotations_data.append((node, rank, len(adjacent_nodes)))
                return annotations_data

            annotations_graph_1 = sorted(get_annotations_data(graph_1), key=lambda x: x[0])
            annotations_graph_2 = sorted(get_annotations_data(graph_2), key=lambda x: x[0])

            # using naive method 
            # to combine two sorted lists
            size_1 = len(annotations_graph_1)
            size_2 = len(annotations_graph_2)
            
            top_n_to_show_label = top_perc_to_show_label * max(size_1, size_2)

            node_text_dict = dict()
            annotations = []
            i, j = 0, 0
            
            while i < size_1 and j < size_2:
                if annotations_graph_1[i][0] < annotations_graph_2[j][0]:
                    node = annotations_graph_1[i][0]
                    len_adjacent_nodes_graph_1 = annotations_graph_1[i][2]
                    if annotations_graph_1[i][2] > min_node_degree_to_show_label and annotations_graph_1[i][1] <= top_n_to_show_label:
                        annotations.append(dict(x=graph_1.nodes[node]['pos'][0], 
                                                y=graph_1.nodes[node]['pos'][1],
                                                text=f"{graph_1.nodes[node]['ticker']}",#<br>{label_graph_1} #{len_adjacent_nodes_graph_1}",
                                                yshift=5 if layout in ['planar'] else 15,
                                                showarrow=True if layout in ['planar'] else False,
                                                arrowhead=1
                                                )
                                            )
                    node_text_dict[node] = f"{graph_1.nodes[node]['ticker']}<br>{label_graph_1} #{len_adjacent_nodes_graph_1}"
                    i += 1
                elif annotations_graph_1[i][0] == annotations_graph_2[j][0]:
                    node = annotations_graph_1[i][0]
                    len_adjacent_nodes_graph_1 = annotations_graph_1[i][2]
                    len_adjacent_nodes_graph_2 = annotations_graph_2[j][2]
                    if ((annotations_graph_1[i][2] > min_node_degree_to_show_label and annotations_graph_1[i][1] <= top_n_to_show_label) or \
                        (annotations_graph_2[j][2] > min_node_degree_to_show_label and annotations_graph_1[j][2] <= top_n_to_show_label)):

                        
                        annotations.append(dict(x=graph_1.nodes[node]['pos'][0], 
                                                y=graph_1.nodes[node]['pos'][1],
                                                text=f"{graph_1.nodes[node]['ticker']}",#<br>{label_graph_1} #{len_adjacent_nodes_graph_1}<br>{label_graph_2} #{len_adjacent_nodes_graph_2}",
                                                yshift=5 if layout in ['planar'] else 15,
                                                showarrow=True if layout in ['planar'] else False,
                                                arrowhead=1
                                                )
                                            )
                    node_text_dict[node] = f"{graph_1.nodes[node]['ticker']}<br>{label_graph_1} #{len_adjacent_nodes_graph_1}<br>{label_graph_2} #{len_adjacent_nodes_graph_2}"
                    i += 1
                    j += 1
                elif annotations_graph_1[i][0] > annotations_graph_2[j][0]:
                    node = annotations_graph_2[j][0]
                    len_adjacent_nodes_graph_2 = annotations_graph_2[j][2]
                    if annotations_graph_2[j][2] > min_node_degree_to_show_label and annotations_graph_1[j][2] <= top_n_to_show_label:
                        annotations.append(dict(x=graph_2.nodes[node]['pos'][0], 
                                                y=graph_2.nodes[node]['pos'][1],
                                                text=f"{graph_2.nodes[node]['ticker']}",#<br>{label_graph_2} #{len_adjacent_nodes_graph_2}",
                                                yshift=5 if layout in ['planar'] else 15,
                                                showarrow=True if layout in ['planar'] else False,
                                                arrowhead=1
                                                )
                                            )
                    node_text_dict[node] = f"{graph_2.nodes[node]['ticker']}<br>{label_graph_2} #{len_adjacent_nodes_graph_2}"
                    j += 1

            while i < size_1:
                node = annotations_graph_1[i][0]
                len_adjacent_nodes_graph_1 = annotations_graph_1[i][2]
                if annotations_graph_1[i][2] > min_node_degree_to_show_label and annotations_graph_1[i][1] <= top_n_to_show_label:
                    annotations.append(dict(x=graph_1.nodes[node]['pos'][0], 
                                            y=graph_1.nodes[node]['pos'][1],
                                            text=f"{graph_1.nodes[node]['ticker']}",#<br>{label_graph_1} #{len_adjacent_nodes_graph_1}",
                                            yshift=5 if layout in ['planar'] else 15,
                                            showarrow=True if layout in ['planar'] else False,
                                            arrowhead=1
                                            )
                                        )
                node_text_dict[node] = f"{graph_1.nodes[node]['ticker']}<br>{label_graph_1} #{len_adjacent_nodes_graph_1}"
                i += 1

            while j < size_2:
                node = annotations_graph_2[j][0]
                len_adjacent_nodes_graph_2 = annotations_graph_2[j][2]
                if annotations_graph_2[j][2] > min_node_degree_to_show_label and annotations_graph_1[j][2] <= top_n_to_show_label:
                    annotations.append(dict(x=graph_2.nodes[node]['pos'][0], 
                                            y=graph_2.nodes[node]['pos'][1],
                                            text=f"{graph_2.nodes[node]['ticker']}",#<br>{label_graph_2} #{len_adjacent_nodes_graph_2}",
                                            yshift=5 if layout in ['planar'] else 15,
                                            showarrow=True if layout in ['planar'] else False,
                                            arrowhead=1
                                            )
                                        )
                node_text_dict[node] = f"{graph_2.nodes[node]['ticker']}<br>{label_graph_2} #{len_adjacent_nodes_graph_2}"
                j += 1

            return annotations, node_text_dict

        if graphs_2:
            annotations, node_text_dict = get_annotation_and_node_text(graph_1, graph_2)
        
            for node in graph_1.nodes():
                node_text.append(node_text_dict[node])
            
            annotations_list.append(annotations)
        else:
            annotations_list.append(get_annotations(graph_1))

        node_trace.text = node_text

        traces.append(node_trace)
        traces.append(edge_trace)
        if graphs_2:
            traces.append(edge_trace_mst)

        titles.append(f"{label_graph_1} and {label_graph_2} at {time}" if graphs_2 else f"{label_graph_1} at {time}")

        slider_ticks.append(f"{time[:10]}")

    fig = go.Figure(data=traces,#[edge_trace, node_trace],
                    layout=go.Layout(
                    title=titles[0],
                    titlefont_size=20,
                    showlegend=True,
                    legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    width=1200, height=600
                    )
                   )

    traces_per_plot = int(len(traces) / len(annotations_list))
    fig.data[0].visible = True
    fig.data[1].visible = True
    if graphs_2:
        fig.data[2].visible = True
    for annotation in annotations_list[0]:
        fig.add_annotation(annotation)

    # Create and add slider
    steps = []
    number_of_steps = int(len(fig.data) / traces_per_plot)
    for i in range(number_of_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"annotations": annotations_list[i],
                    "title": titles[i]},
                ],  # layout attribute
            label=slider_ticks[i]
        )
        step["args"][0]["visible"][traces_per_plot*i] = True  # Toggle i'th trace to "visible"
        step["args"][0]["visible"][traces_per_plot*i+1] = True  # Toggle i'th trace to "visible"
        if graphs_2:
            step["args"][0]["visible"][traces_per_plot*i+2] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Time: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )

    fig.show()