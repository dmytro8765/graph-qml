import random

import networkx as nx
import numpy as np
import torch

dataset = []
num_samples = 3000
type_graph = 0 
qubits = 6
k = 3

def contains_clique(graph):
    return any(len(clique) == k for clique in nx.enumerate_all_cliques(graph))

for _ in range(num_samples):
    p = random.uniform(0.1, 0.9)
    
    if type_graph == 0:
        # Generate graph that CONTAINS at least one 3-clique
        g = nx.gnp_random_graph(qubits, p)
        while not contains_clique(g):
            p = random.uniform(0.15, 0.45)
            g = nx.gnp_random_graph(qubits, p)
        label = 1
    else:
        # Generate graph that DOES NOT contain any 3-cliques
        g = nx.gnp_random_graph(qubits, p)
        while contains_clique(g):
            p = random.uniform(0.15, 0.45)
            g = nx.gnp_random_graph(qubits, p)
        label = 0

    array = np.append(nx.to_numpy_array(g).flatten(), label)
    dataset.append(array)

    type_graph = (type_graph + 1) % 2  # Alternate

dataset = torch.tensor(np.array(dataset))
torch.save(dataset, "/Users/home/qiskit_env/Pennylane/data/max_clique/nodes_6-graphs_3000.pt")
