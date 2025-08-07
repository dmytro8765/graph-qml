"""
Generate a dataset, whose targets DO NOT respect possible isomorphisms.

Create main graphs and subgraphs, such that:
- 50% -> subgraph is contained in the big graph;
- 50% -> subgraph is not contained.

Label for each node: 1, if contained in the found subgraph, 0 otherwise.
"""

import networkx as nx
import numpy as np
import torch
import random

dataset = []
num_samples = 3000
main_nodes = 6
sub_nodes = 4

for _ in range(num_samples):
    while True:
        p_big = random.uniform(0.3, 0.9)
        g_big = nx.gnp_random_graph(main_nodes, p_big)
        if nx.is_connected(g_big):
            break

    big_array = nx.to_numpy_array(g_big).flatten()

    is_positive = random.random() < 0.5

    # random in [0, 1] < 0.5 --> is_positive = True --> generate positive sample
    # random in [0, 1] >= 0.5 --> is_positive = False --> generate negative sample

    if is_positive:  # generate subgraph, contained in the main graph
        while True:
            p_small = random.uniform(0.3, 0.9)
            g_small = nx.gnp_random_graph(sub_nodes, p_small)

            if nx.is_connected(g_small):
                GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
                iso = 0
                for mapping in GM.subgraph_isomorphisms_iter():
                    label = [0] * main_nodes
                    print("Mapping keys (main graph nodes):", list(mapping.keys()))
                    print("Mapping values (subgraph nodes):", list(mapping.values()))
                    for big_node in mapping.keys():
                        label[big_node] = 1
                    break
                if 1 in label:
                    break
        small_array = nx.to_numpy_array(g_small).flatten()
    else:  # generate subgraph, NOT contained in the main graph
        while True:
            label = [0] * main_nodes
            p_small = random.uniform(0.3, 0.9)
            g_small = nx.gnp_random_graph(sub_nodes, p_small)

            if nx.is_connected(g_small):
                GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
                if not GM.subgraph_is_isomorphic():
                    break
        small_array = nx.to_numpy_array(g_small).flatten()

    print("Final label for the datapoint: ", label)
    print()
    print()

    # Combine main and sub graphs into 1 adjaceny matrix, by filliing out the inter-graph connections with 0s.
    big_adj = nx.to_numpy_array(g_big)  # shape (6,6)
    small_adj = nx.to_numpy_array(g_small)  # shape (4,4)

    combined_adj = np.block(
        [[big_adj, np.zeros((6, 4))], [np.zeros((4, 6)), small_adj]]
    )
    # print(combined_adj)

    data_point = np.concatenate([big_array, small_array, combined_adj.flatten(), label])

    # print(data_point)

    dataset.append(data_point)
    print()
    print()

dataset = torch.tensor(np.array(dataset), dtype=torch.float)
torch.save(
    dataset,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph/graph-6_subgraph-4_per_qubit_3000.pt",
)
