"""
Generate a dataset, whose targets respect possible isomorphisms.

Create main graphs and subgraphs, such that:
- 50% -> subgraph is contained in the big graph
(label = all possible locations of the subgraph in the big graph).
- 50% -> subgraph is not contained (label = [0, 0, 0, 0, 0, 0], for 6 (or more) qubit main graph).

"""

import networkx as nx
import numpy as np
import torch
import random

dataset = []
final_dataset = []
num_samples = 3000
main_nodes = 6
sub_nodes = 4

longest_label = 0
iso_lengths = []

k = 0  # determines if current subgraph is contained in the main graph or not

for _ in range(num_samples):

    # 1. Create the main (connected) graph:
    while True:
        p_big = random.uniform(0.3, 0.9)
        g_big = nx.gnp_random_graph(main_nodes, p_big)
        if nx.is_connected(g_big):
            break

    big_array = nx.to_numpy_array(g_big).flatten()

    # Future list of all configurations of [subgraph contained in the main graph]:
    isomorphisms = []

    print("k = ", k)

    if k % 2 == 0:  # generate subgraph, contained in the main graph

        while True:
            # Generate a subgraph
            p_small = random.uniform(0.3, 0.9)
            g_small = nx.gnp_random_graph(sub_nodes, p_small)

            # Connected -> go through all iso, contained in the main graph, and add to the list of all iso
            # (or go through the loop until an iso, contained in main, is found):
            if nx.is_connected(g_small):
                GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
                if GM.subgraph_is_isomorphic():
                    for mapping in GM.subgraph_isomorphisms_iter():
                        label = [0] * main_nodes
                        # print("Mapping keys (main graph nodes):", list(mapping.keys()))
                        # print("Mapping values (subgraph nodes):", list(mapping.values()))
                        for big_node in mapping.keys():
                            label[big_node] = 1

                        isomorphisms.append(
                            label
                        )  # add newly found mapping to the total list

                    break

        # Save the subgraph:
        small_array = nx.to_numpy_array(g_small).flatten()
        k += 1

    else:  # generate subgraph, NOT contained in the main graph

        while True:
            # Generate a subgraph
            p_small = random.uniform(0.3, 0.9)
            g_small = nx.gnp_random_graph(sub_nodes, p_small)

            # Connected -> check if contained; stop the search if NOT contained
            if nx.is_connected(g_small):
                GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
                if not GM.subgraph_is_isomorphic():
                    break

        # Save the subgraph:
        small_array = nx.to_numpy_array(g_small).flatten()
        k += 1

    # Combine main and sub graphs into one adjacency matrix
    # by filling out the inter-graph connections with 0s.
    big_adj = nx.to_numpy_array(g_big)  # shape (main_nodes, main_nodes)
    small_adj = nx.to_numpy_array(g_small)  # shape (sub_nodes, sub_nodes)

    combined_adj = np.block(
        [[big_adj, np.zeros((6, 4))], [np.zeros((4, 6)), small_adj]]
    )

    # Combine separate graphs and their combinations into one datapoint:
    data_point = np.concatenate([big_array, small_array, combined_adj.flatten()])

    # Find the longest lable currently:
    if len(isomorphisms) > longest_label:
        longest_label = len(isomorphisms)

    # Save lengths of each label with iso, to know how much to fill out:
    iso_lengths.append(len(isomorphisms))

    # Add all the found iso at the end of the datapoint to be label:
    final_data_point = np.append(data_point, isomorphisms)
    dataset.append(final_data_point)

"""
So far:
1) generated all big, small graphs where even are contained, odd are NOT;
2) contained -> added the label with all isos.
"""

print("Subgraphs generated: ", k)
print("Longest label: ", longest_label)

i = 0  # keep track of the current label length with iso_lengths[i]

# Each datapoint: fill out the label with (0, 0, 0, 0, 0, 0) as often as the longest label requires.
for data_point in dataset:
    final_data_point = np.append(
        data_point, [0.0] * main_nodes * (longest_label - iso_lengths[i])
    )
    final_dataset.append(final_data_point)
    print(final_data_point)
    i += 1

final_dataset = torch.tensor(np.array(final_dataset), dtype=torch.float)
torch.save(
    final_dataset,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph/graph-6_subgraph-4_per_qubit_3000_isomorph.pt",
)

"""
Final datapoint:
- 36 elements -> main graph adjacency matrix;
- 16 elements -> subgraph adjacency matrix;
- 100 elements -> combined adjacency of main and sub;
- remainder -> label.
"""
