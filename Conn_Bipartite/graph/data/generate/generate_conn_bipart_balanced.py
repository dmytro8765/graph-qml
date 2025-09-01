import random

import networkx as nx
import numpy as np
import torch

dataset = []
num_samples = 3000
type_graph = 0
type_graph_half = 0
qubits = 8

for _ in range(num_samples):
    p = random.uniform(0.15, 0.45)
    ## --> otherwise dataset biased towards being not bipartite and connected (with [0.3, 0.6])
    ## --> bipartite: 0.90 - 0.05
    ## --> connected: 0.05 - 0.90
    n = random.randint(2, 6)

    if type_graph == 0:
        g = nx.algorithms.bipartite.generators.random_graph(qubits - n, n, p)
        while not nx.is_connected(g):
            p = random.uniform(0.15, 0.45)
            g = nx.algorithms.bipartite.generators.random_graph(qubits - n, n, p)
        array = np.append(nx.to_numpy_array(g).flatten(), 1)
    else:
        if type_graph_half == 0:
            g = nx.algorithms.bipartite.generators.random_graph(qubits - n, n, p)
            while nx.is_connected(g):
                p = random.uniform(0.15, 0.45)
                g = nx.algorithms.bipartite.generators.random_graph(qubits - n, n, p)
            array = np.append(nx.to_numpy_array(g).flatten(), 0)
        else:
            if type_graph_half == 1:
                g = nx.gnp_random_graph(qubits, p)
                while nx.algorithms.bipartite.is_bipartite(g) and not nx.is_connected(
                    g
                ):
                    p = random.uniform(0.15, 0.45)
                    g = nx.gnp_random_graph(qubits, p)
                array = np.append(nx.to_numpy_array(g).flatten(), 0)
            else:
                g = nx.gnp_random_graph(qubits, p)
                while nx.algorithms.bipartite.is_bipartite(g) and nx.is_connected(g):
                    p = random.uniform(0.15, 0.45)
                    g = nx.gnp_random_graph(qubits, p)
                array = np.append(nx.to_numpy_array(g).flatten(), 0)
        type_graph_half = (type_graph_half + 1) % 3

    type_graph = (type_graph + 1) % 2
    dataset.append(array)
dataset = torch.tensor(np.array(dataset))
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Conn_Bipartite/graph/data/datasets/nodes_8-graphs_3000_balanced.pt",
)
