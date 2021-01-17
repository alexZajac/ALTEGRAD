"""
Graph Mining - ALTEGRAD - Jan 2021
"""

import networkx as nx
import numpy as np
import torch
from random import randint
from networkx.generators.random_graphs import fast_gnp_random_graph


def create_dataset():
    Gs = list()
    y = list()

    # Task 6

    ##################
    # your code here #
    n_graphs = 50
    n = 10
    for _ in range(n_graphs):
        class_1_graph = fast_gnp_random_graph(n, 0.2)
        class_2_graph = fast_gnp_random_graph(n, 0.4)
        Gs.extend([class_1_graph, class_2_graph])
        y.extend([0, 1])
    ##################

    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
