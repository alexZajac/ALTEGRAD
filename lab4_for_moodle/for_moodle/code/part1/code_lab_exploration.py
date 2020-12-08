"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


############## Task 1

##################
data_path = Path('..') / 'datasets' / 'CA-HepTh.txt'
graph = nx.readwrite.edgelist.read_edgelist(data_path, delimiter='\t')
num_nodes = graph.number_of_nodes()
num_edges = graph.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")
##################



############## Task 2

##################
cc_generator = nx.connected_components(graph)
sorted_components = [c for c in sorted(cc_generator, key=len, reverse=True)]
print(f"Number of connected components: {len(sorted_components)}")
largest_cc = sorted_components[0]
largest_subgraph = graph.subgraph(largest_cc).copy()
num_nodes_largest_cc = largest_subgraph.number_of_nodes()
num_edges_largest_cc = largest_subgraph.number_of_edges()
print(f"Number of nodes of largest connected component: {num_nodes_largest_cc}")
print(f"Number of edges of largest connected component: {num_edges_largest_cc}")
print(f"Fraction of nodes of total graph: {num_nodes_largest_cc / num_nodes}")
print(f"Fraction of edges of total graph: {num_edges_largest_cc / num_edges}")
##################



############## Task 3
# Degree
degree_sequence = [graph.degree(node) for node in graph.nodes()]


##################
print(f"Min of degrees of the graph: {np.min(degree_sequence)}")
print(f"Max of degrees of the graph: {np.max(degree_sequence)}")
print(f"Average of degrees of the graph: {np.mean(degree_sequence)}")
print(f"Median of degrees of the graph: {np.median(degree_sequence)}")
##################



############## Task 4

##################
p  = Path('.') / 'degree_histogram.png'
hist_degrees = nx.degree_histogram(graph)
degrees = range(len(hist_degrees))
plt.loglog(degrees, hist_degrees)
plt.xlabel("Degree")
plt.ylabel("frequency")
plt.show()
plt.savefig(p)
##################




############## Task 5

##################
print(f"Global clustering coefficient: {nx.transitivity(graph)}")
##################