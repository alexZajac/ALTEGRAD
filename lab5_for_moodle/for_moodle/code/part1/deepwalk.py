"""
Deep Learning on Graphs - ALTEGRAD - Dec 2020
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec


# Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):

    ##################
    # your code here #
    walk = [node]
    for _ in range(walk_length-1):
        neighbors = list(G.neighbors(walk[-1]))
        random_neighbor = neighbors[randint(0, len(neighbors)-1)]
        walk.append(random_neighbor)
    ##################

    walk = [str(node) for node in walk]
    return walk


# Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []

    ##################
    # your code here #
    nodes = list(G.nodes())
    for _ in range(num_walks):
        permuted_nodes = np.random.permutation(nodes)
        for node in permuted_nodes:
            walks.append(random_walk(G, node, walk_length))
    ##################

    return walks

# Simulates walks and uses the Skipgram model to learn node representations


def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model
