"""
Deep Learning on Graphs - ALTEGRAD - Jan 2021
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn


def normalize_adjacency(A):
    # Task 1

    ##################
    A_tilde = A + sp.identity(A.shape[0])
    D_tilde = sp.diags(A_tilde.sum(axis=1).A1)
    D_pow_tilde = D_tilde.power(-1)
    A_normalized = D_pow_tilde @ A_tilde
    ##################

    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    # Task 3

    ##################
    # your code here #
    z_pred = torch.mm(z, z.T)
    y, y_pred = [], []

    # non zero elements
    idx = adj._indices()
    y_pred.append(z_pred[idx[0, :], idx[1, :]])
    y.append(torch.ones(idx.size(1)).to(device))

    # zeros randomly sampled elements
    rnd_idx = torch.randint(z_pred.size(0), idx.size())
    y_pred.append(z_pred[rnd_idx[0, :], rnd_idx[1, :]])
    y.append(torch.zeros(rnd_idx.size(1)).to(device))

    # concat tensors
    y_pred = torch.cat(y_pred, dim=0)
    y = torch.cat(y, dim=0)
    ##################

    loss = mse_loss(y_pred, y)
    return loss
