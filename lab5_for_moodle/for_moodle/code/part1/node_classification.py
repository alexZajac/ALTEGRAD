"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk
import matplotlib.pyplot as plt

# Loads the karate network
G = nx.read_weighted_edgelist(
    '../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt',
                          delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


# Task 5

##################
pos = nx.spring_layout(G)
node_color = ["#fdb827" if label else "#21209c" for label in y]
nx.draw_networkx(G, pos, node_color=node_color)
plt.savefig("karate_network.png")
##################


# Task 6
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)  # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]


# Task 7

##################
clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with the deepwalk embeddings: {accuracy:.3f}")
##################


# Task 8

##################
# your code here #
A = nx.to_numpy_matrix(G)
spectral_embeddings = SpectralEmbedding(
    affinity="precomputed", n_components=1, n_jobs=-1
).fit_transform(A)

X_train = spectral_embeddings[idx_train, :]
X_test = spectral_embeddings[idx_test, :]

clf = LogisticRegression().fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with the spectral embeddings: {accuracy:.3f}")
##################
