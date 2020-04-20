import itertools

from bokeh.models import Circle, Plot, HoverTool
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.palettes import Category20_20 as palette
import networkx as nx
import numpy as np
import torch
from torch.nn import functional as F

from lib import GraphEmbedding


colors = itertools.cycle(palette)

    
def generate_networkx_graph(emb: GraphEmbedding, vertex_labels:list, 
                            edge_probability_threshold:float = 0.5):
    G = nx.Graph()

    # extract vertices and edges
    from_ix, to_ix = emb.edge_sources, emb.edge_targets
    weights = F.softplus(emb.edge_weight_logits).view(-1).data.numpy()
    mean_weight = weights[1:].mean()
    num_vertices, num_edges = len(emb.slices) - 1, len(from_ix)
    edge_probabilities = torch.sigmoid(emb.edge_adjacency_logits.view(-1)).data.numpy()
    edge_exists = edge_probabilities >= edge_probability_threshold
    assert num_vertices == len(vertex_labels)

    # handle vertices
    G.add_nodes_from(list(range(num_vertices)))
    nx.set_node_attributes(G, {k:v for k, v in enumerate(vertex_labels)}, 'label')

    # handle weighted edges, skip first loop edge
    edges = [(from_ix[i], to_ix[i], weights[i]) for i in range(1, num_edges) if edge_exists[i]]
    G.add_weighted_edges_from(edges)

    return G


def draw_networkx_graph(G, dataset_name: str, weighted:False, cmap_name:str = 'Spectral32'):
    # dataset_name: Displayed in titl
    # weighted: if True, renders edge's width proportional to weight value
    # cmap_name: Bokeh palette name

    vertex_labels = [y['label'] for x,y in G.nodes(data=True)]
    unique_labels = list(set(vertex_labels))
    weights = [data['weight'] for node1, node2, data in G.edges(data=True)]

    plot = figure(title=f"{dataset_name.capitalize()} Prodige Graph", plot_width=600, plot_height=400, 
                  x_range=(-1.1,1.1), y_range=(-1.1,1.1))
    layout = nx.spring_layout(G, k=1.1/(G.number_of_nodes()**0.5), iterations=100)
    
    nodes, nodes_coordinates = zip(*sorted(layout.items()))
    nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
    color = [palette[i] for i in vertex_labels]
    node_degree = [v for k,v in G.degree()]

    nodes_source = ColumnDataSource(dict(x=nodes_xs, y=nodes_ys, name=nodes, 
                                         label=vertex_labels, 
                                         degree=node_degree, color=color))
    r_circles = plot.circle('x', 'y', source=nodes_source, name= "Node_list", line_color=None, 
                            size=10, fill_color="color", level = 'overlay') 
    hover1 = HoverTool(tooltips=[('Node ID', '@name'),('Label','@label'), ('Degree', '@degree')],
    renderers=[r_circles])

    lines_source = ColumnDataSource(_get_edges_specs(G, layout))    
    r_lines = plot.multi_line('xs', 'ys', line_width='line_width', 
                              color='navy', source=lines_source) 
    hover2 = HoverTool(tooltips=[('Node 1','@from_node'), ('Node 2','@to_node'), ('Weight','@weight')], renderers=[r_lines])

    plot.tools.append(hover1)
    plot.tools.append(hover2)

    return plot


def _get_edges_specs(_network, _layout): 
    d = dict(xs=[], ys=[], line_width=[],from_node=[],to_node=[], alpha=[])
    weights = [data['weight'] for u, v, data in _network.edges(data=True)]
    max_weight = max(weights)
    calc_alpha = lambda h: 0.1 + 0.6 * (h / max_weight)
    for u, v, data in _network.edges(data=True):
        d['xs'].append([_layout[u][0], _layout[v][0]])
        d['from_node'].append(u)
        d['to_node'].append(v)
        d['ys'].append([_layout[u][1], _layout[v][1]])
        d['line_width'].append(calc_alpha(data['weight']) * 3)
        d['alpha'].append(calc_alpha(data['weight']))
    d['weight'] = weights
    return d
