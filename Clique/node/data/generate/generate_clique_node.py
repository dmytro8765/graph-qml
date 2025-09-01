"""
Generate a dataset for node classification in a graph.

Label for each node:
- 1: node contained in some k-clique;
- 0: otherwise.

Total label: (0 or 1) * total nodes in the graph.

"""

import random

import networkx as nx
import numpy as np
import torch

dataset = []
num_samples = 3000
type_graph = 0
qubits = 8
k = 4

# 6 nodes => 1 w/o clique 4, 2 w clique 4
# 8 nodes => 1 w/o clique 4, 4 w clique 4
# 10 nodes => all with a clique


def contains_k_clique(graph, k):
    return any(len(clique) >= k for clique in nx.enumerate_all_cliques(graph))


def is_node_in_clique(graph, node, clique_size):
    cliques = list(nx.enumerate_all_cliques(graph))
    for clique in cliques:
        if len(clique) == clique_size and node in clique:
            return True
    return False


for _ in range(num_samples):
    p = random.uniform(0.1, 0.9)

    if type_graph in [0, 1]:
        labels = []
        g = nx.gnp_random_graph(qubits, p)
        while not contains_k_clique(g, k):
            p = random.uniform(0.15, 0.45)
            g = nx.gnp_random_graph(qubits, p)
        for node in g:
            if is_node_in_clique(g, node, k) == True:
                labels = np.append(labels, 1)
            else:
                labels = np.append(labels, 0)
    else:
        labels = []
        g = nx.gnp_random_graph(qubits, p)
        while contains_k_clique(g, k):
            p = random.uniform(0.15, 0.45)
            g = nx.gnp_random_graph(qubits, p)
        for node in g:
            labels = np.append(labels, 0)

    array = nx.to_numpy_array(g).flatten()
    for i in range(len(labels)):
        array = np.append(array, labels[i])

    dataset.append(array)

    type_graph = (type_graph + 1) % 3  # Alternate

dataset = torch.tensor(np.array(dataset))
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Clique/node/data/datasets/nodes_6-graphs_3000_node.pt",
)
