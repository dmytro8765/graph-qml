"""
Generate a dataset, whose targets DO NOT respect possible isomorphisms.

Create main graphs and subgraphs, such that:
- 50% -> subgraph is contained in the big graph;
- 50% -> subgraph is not contained.

Labels: 1 for contained, 0 for not contained. (1 label per graph).
"""

import networkx as nx
import numpy as np
import torch
import random

dataset = []
num_samples = 3000
main_nodes = 6
subgraph_nodes = 4

positive_target = num_samples // 2
negative_target = num_samples - positive_target
positive_count = 0
negative_count = 0

while len(dataset) < num_samples:
    # Generate a connected random 6-node graph
    while True:
        p_big = random.uniform(0.3, 0.9)
        g_big = nx.gnp_random_graph(main_nodes, p_big)
        if nx.is_connected(g_big):
            break
    big_array = nx.to_numpy_array(g_big).flatten()

    # Generate a connected random 4-node graph
    while True:
        p_small = random.uniform(0.3, 0.9)
        g_small = nx.gnp_random_graph(subgraph_nodes, p_small)
        if nx.is_connected(g_small):
            break
    small_array = nx.to_numpy_array(g_small).flatten()

    GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
    is_subgraph = any(GM.subgraph_isomorphisms_iter())

    if is_subgraph and positive_count < positive_target:
        label = [1]
        positive_count += 1
    elif not is_subgraph and negative_count < negative_target:
        label = [0]
        negative_count += 1
    else:
        continue  # skip this sample if quota filled

    # Combine main and sub graphs into 1 adjaceny matrix, by filliing out the inter-graph conenctions with 0s.
    big_adj = nx.to_numpy_array(g_big)  # shape (6,6)
    small_adj = nx.to_numpy_array(g_small)  # shape (4,4)

    combined_adj = np.block(
        [
            [big_adj, np.zeros((main_nodes, subgraph_nodes))],
            [np.zeros((subgraph_nodes, main_nodes)), small_adj],
        ]
    )

    data_point = np.concatenate([big_array, small_array, combined_adj.flatten(), label])
    dataset.append(data_point)

dataset = torch.tensor(np.array(dataset), dtype=torch.float)
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph/graph-6_subgraph-4_3000.pt",
)
