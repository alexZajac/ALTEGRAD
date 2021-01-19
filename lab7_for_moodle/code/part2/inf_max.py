"""
Influence Maximization - ALTEGRAD - Jan 2021
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from utils import load_data


def independent_cascade(G, S, p, mc):

    # Loops over the Monte-Carlo Simulations
    spread = list()
    for i in range(mc):
        # Simulates propagation process
        new_active = S[:]
        A = S[:]
        while new_active:

            # For each newly active node, finds its neighbors that become activated
            new_ones = list()
            for node in new_active:
                # Determines neighbors that become infected
                neighbors = list(G.neighbors(node))
                success = np.random.uniform(0, 1, len(neighbors)) < p
                new_ones += list(np.extract(success, neighbors))

            new_active = list(set(new_ones) - set(A))

            # Adds newly activated nodes to the set of activated nodes
            A += new_active

        # Appends the number of activated nodes
        spread.append(len(A))

    return np.mean(spread)


def greedy_algorithm(G, k, p, mc):

    S = list()
    spread = list()

    # Task 8

    ##################
    for i in range(k):
        print(f"Beginning iteration for node: {i+1}")
        current_nodes = list(G.nodes-set(S))
        current_cascade = independent_cascade(G, S, p, mc)
        current_spread = np.array([
            independent_cascade(G, S + [u], p, mc) - current_cascade
            for u in current_nodes
        ])
        spread.append(np.mean(current_spread))
        v = current_nodes[np.argmax(current_spread)]
        S.append(v)
    ##################

    return S, spread


# Loads network
G = load_data()
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

# Task 9

##################
k, p, mc = 5, 0.1, 100
influencial_nodes, spread = greedy_algorithm(G, k, p, mc)
print(f"Top-{k} influential nodes: {influencial_nodes}")
##################

# Visualizes the spread with respect to the size of the set
plt.plot(range(1, len(spread)+1), spread)
plt.xlabel('size of set')
plt.ylabel('expected spread')
plt.show()
plt.savefig("expected_spread_each_set_size.png")
