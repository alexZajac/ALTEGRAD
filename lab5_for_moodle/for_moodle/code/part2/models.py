"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    """Simple GNN model"""

    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        # Tasks 10 and 13

        ##################
        # your code here #
        Z_0 = self.relu(torch.mm(adj, self.fc1(x_in)))
        Z_0 = self.dropout(Z_0)
        Z_1 = self.relu(torch.mm(adj, self.fc2(Z_0)))
        x = self.fc3(Z_1)
        ##################

        return F.log_softmax(x, dim=1), Z_1
