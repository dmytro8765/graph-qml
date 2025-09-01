"""
Generate datasets for Hamiltonian path/cycle detection.
"""

import random

import networkx as nx
import numpy as np
import torch


def hamilton_path(G):
    F = [(G, [list(G.nodes())[0]])]
    n = G.number_of_nodes()
    while F:
        graph, path = F.pop()
        confs = []
        neighbors = (
            node for node in graph.neighbors(path[-1]) if node != path[-1]
        )  # Exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g, conf_p))
        for g, p in confs:
            if len(p) == n:
                return p
            F.append((g, p))
    return None


def hamilton_cycle(G):
    F = [(G, [list(G.nodes())[0]])]  # Start from any node
    n = G.number_of_nodes()

    while F:
        graph, path = F.pop()
        confs = []

        # Get neighbors of the last node in the current path
        neighbors = (
            node for node in graph.neighbors(path[-1]) if node not in path
        )  # Exclude self-loops and already visited nodes

        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            confs.append((graph, conf_p))  # Keep the original graph intact

        for g, p in confs:
            if len(p) == n:
                # Check if the last node is connected to the first node to form a cycle
                if p[-1] in g.neighbors(p[0]):
                    return p  # This is a Hamiltonian cycle
            F.append((g, p))

    return None


dataset = []
num_samples = 3000
type_graph = 0
qubits = 8

for _ in range(num_samples):
    p = random.uniform(0.3, 0.9)

    if type_graph == 0:
        g = nx.gnp_random_graph(qubits, p)
        while hamilton_cycle(g) is None:
            p = random.uniform(0.3, 0.9)
            g = nx.gnp_random_graph(qubits, p)
        array = np.append(nx.to_numpy_array(g).flatten(), 1)
    else:
        g = nx.gnp_random_graph(qubits, p)
        while hamilton_cycle(g) is not None:
            p = random.uniform(0.3, 0.9)
            g = nx.gnp_random_graph(qubits, p)
        array = np.append(nx.to_numpy_array(g).flatten(), 0)
    type_graph = (type_graph + 1) % 2
    dataset.append(array)

print(dataset[:4])

dataset = torch.tensor(np.array(dataset))
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Hamiltonian/graph/data/datasets/nodes_8-graphs_3000.pt",
)
