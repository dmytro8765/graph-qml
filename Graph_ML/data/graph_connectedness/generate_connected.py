import random

import networkx as nx
import numpy as np
import torch

dataset = []
num_samples = 3000
type_graph = 0
qubits = 4
for _ in range(num_samples):
    p = random.uniform(0.3, 0.9)
    n = random.randint(2, 6)

    if type_graph == 0:
        g = nx.gnp_random_graph(qubits, p)
        while not nx.is_connected(g):
            p = random.uniform(0.3, 0.9)
            g = nx.gnp_random_graph(qubits, p)
        array = np.append(nx.to_numpy_array(g).flatten(), 1)
    else:
        g = nx.gnp_random_graph(qubits, p)
        while nx.is_connected(g):
            p = random.uniform(0.3, 0.9)
            g = nx.gnp_random_graph(qubits, p)
        array = np.append(nx.to_numpy_array(g).flatten(), 0)
    type_graph = (type_graph + 1) % 2
    dataset.append(array)
dataset = torch.tensor(np.array(dataset))
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/graph_connectedness/nodes_4-graphs_3000.pt",
)
