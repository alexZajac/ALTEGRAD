"""
Graph Mining - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
from random import randint
from sklearn.cluster import KMeans
from pathlib import Path
from collections import Counter


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    A = nx.adjacency_matrix(G)
    inv_degree_sequence = [1 / G.degree(node) for node in G.nodes()]
    D_inv = diags(inv_degree_sequence)
    n = G.number_of_nodes()
    Lrw = np.eye(n) - D_inv @ A
    eig_values, eig_vectors = eigs(Lrw, k=k, which='SR')
    eig_vectors = eig_vectors.real
    kmean = KMeans(n_clusters=k)
    kmean.fit(eig_vectors)
    clustering = {node: kmean.labels_[i] for i, node in enumerate(G.nodes())}
    ##################
    return clustering



############## Task 7

##################
data_path = Path('..') / 'datasets' / 'CA-HepTh.txt'
graph = nx.readwrite.edgelist.read_edgelist(data_path, delimiter='\t')
k = 50
max_component = max(nx.connected_components(graph), key=len)
giant_component = graph.subgraph(max_component)
clustering = spectral_clustering(giant_component, k)
print(f"Clusters: {clustering}")
print(f"Number of nodes per cluster: {Counter(clustering.values())}")
##################



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    ##################
    m = G.number_of_edges()
    clusters = set(clustering.values())
    modularity = 0

    for cluster in clusters:
        nodes_in_cluster = [
            node for node in G.nodes() if clustering[node] == cluster
        ]
        current_subgraph = G.subgraph(nodes_in_cluster)
        lc = current_subgraph.number_of_edges()

        dc = sum(G.degree(node) for node in nodes_in_cluster)
        modularity += lc/m - (dc/(2*m))**2
    ##################
    return modularity



############## Task 9

##################
print(f"Modularity for the giant component: {modularity(giant_component, clustering)}")
random_clustering = {node: randint(0, k-1) for node in giant_component.nodes()}
print(f"Modularity for a random component: {modularity(giant_component, random_clustering)}")
##################