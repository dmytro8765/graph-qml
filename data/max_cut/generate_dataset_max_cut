import random
import networkx as nx
import numpy as np
import torch
import itertools
import pandas as pd

dataset = []
dataset_plot = []
num_samples = 3000
qubits = 6

# Brute-force Max-Cut for small graphs
def max_cut_brute_force(G):
    nodes = list(G.nodes())
    n = len(nodes)
    best_cut = {}
    max_cut_value = -1

    for bits in itertools.product([0, 1], repeat=n):
        cut_value = 0
        group = {nodes[i]: bits[i] for i in range(n)}

        for u, v in G.edges():
            if group[u] != group[v]:
                cut_value += 1

        if cut_value > max_cut_value:
            max_cut_value = cut_value
            best_cut = group.copy()

    return best_cut, max_cut_value

for _ in range(num_samples):
    # Generate a connected random graph
    p = random.uniform(0.2, 0.6)
    g = nx.gnp_random_graph(qubits, p)
    while not nx.is_connected(g):
        p = random.uniform(0.2, 0.6)
        g = nx.gnp_random_graph(qubits, p)

    # Get max-cut partitioning
    cut_assignment, cut_value = max_cut_brute_force(g)

    # Create label array based on node group (0 or 1)
    labels = np.array([cut_assignment[node] for node in g.nodes()])

    # Flatten adjacency matrix + append labels at the end
    array = nx.to_numpy_array(g).flatten()
    dataset_plot.append(array)

    array = np.append(array, cut_value)
    array = np.append(array, labels)
    dataset.append(array)

dataframe_adjecency = pd.DataFrame({'Adjacencies': dataset_plot})
dataframe_adjecency.to_csv("/Users/home/qiskit_env/Pennylane/data/max_cut/nodes_6-graphs_3000_per_qubit_ADJACENCY.csv", index=False)

# Save as tensor
print(dataset[0])
dataset = torch.tensor(np.array(dataset), dtype=torch.float32)

label_counts = np.zeros((qubits, 2))  # shape: (6 nodes, 2 label counts)

for sample in dataset.numpy():
    labels = sample[-qubits:]  # last 6 elements are labels
    for i, label in enumerate(labels):
        label_counts[i, int(label)] += 1

torch.save(dataset, "/Users/home/qiskit_env/Pennylane/data/max_cut/nodes_6-graphs_3000_per_qubit.pt")


