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
main_nodes = 8
sub_nodes = 5

"""
Generate 2 datasets:
1. Label contains ALL isomorphisms, to compute the accuracy after teaining.
2. Label contains average value per node, to perform the training process itself faster.
"""

""" 1. Dataset for accuracy, containing all isomorphisms. """
longest_label = 0
iso_lengths = []

k = 0  # determines if current subgraph is contained in the main graph or not
graphs_generated = 0

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

    # 2. Find a contained/uncontained subgraph:

    if k % 2 == 0:  # generate subgraph, contained in the main graph

        # pick a subgraph inside the big graph, to guarantee its contained
        nodes = random.sample(list(g_big.nodes()), sub_nodes)
        g_small = g_big.subgraph(nodes).copy()

        GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
        if GM.subgraph_is_isomorphic():
            for mapping in GM.subgraph_isomorphisms_iter():
                label = [0] * main_nodes
                # print("Mapping keys (main graph nodes):", list(mapping.keys()))
                # print("Mapping values (subgraph nodes):", list(mapping.values()))
                for big_node in mapping.keys():
                    label[big_node] = 1

                isomorphisms.append(label)  # add newly found mapping to the total list

        # Save the subgraph:
        small_array = nx.to_numpy_array(g_small).flatten()
        k += 1
        graphs_generated += 1

    else:
        if (
            len(list(g_big.edges())) == main_nodes * (main_nodes - 1) / 2
        ):  # main graph complete: still only pick CONTAINED subgraph

            nodes = random.sample(list(g_big.nodes()), sub_nodes)
            g_small = g_big.subgraph(nodes).copy()

            GM = nx.algorithms.isomorphism.GraphMatcher(g_big, g_small)
            for mapping in GM.subgraph_isomorphisms_iter():
                label = [0] * main_nodes
                for big_node in mapping.keys():
                    label[big_node] = 1

                isomorphisms.append(label)  # add newly found mapping to the total list

            # Save the subgraph and DON'T increment the k counter,
            # so that we still create a negative datapoint afterwards:
            small_array = nx.to_numpy_array(g_small).flatten()
            graphs_generated += 1

        else:  # finally generate subgraph, NOT contained in the main graph
            while True:
                # pick a subgraph inside the big graph, to guarantee its contained
                nodes = random.sample(list(g_big.nodes()), sub_nodes)
                g_small = g_big.subgraph(nodes).copy()
                if len(list(g_small.edges())) != sub_nodes * (sub_nodes - 1) / 2:
                    # ADD an edge that hasn't existed prior, if the subgraph was not complete
                    # otherwise: cannot add an edge, and will never leave the loop
                    while True:
                        u, v = random.sample(list(g_small.nodes()), 2)
                        if not g_small.has_edge(u, v):
                            g_small.add_edge(u, v)
                            break
                    break

            # Save the subgraph:
            small_array = nx.to_numpy_array(g_small).flatten()
            k += 1
            graphs_generated += 1

    # Combine main and sub graphs into one adjacency matrix
    # by filling out the inter-graph connections with 0s.
    big_adj = nx.to_numpy_array(g_big)  # shape (main_nodes, main_nodes)
    small_adj = nx.to_numpy_array(g_small)  # shape (sub_nodes, sub_nodes)

    combined_adj = np.block(
        [
            [big_adj, np.zeros((main_nodes, sub_nodes))],
            [np.zeros((sub_nodes, main_nodes)), small_adj],
        ]
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

print("Subgraphs generated: ", graphs_generated)
print("Longest label: ", longest_label)

i = 0  # keep track of the current label length with iso_lengths[i]

balance = 0

# Each datapoint: fill out the label with (0 * main_nodes) as often as the longest label requires.
for data_point in dataset:
    if iso_lengths[i] == 0:
        balance += 1
    final_data_point = np.append(
        data_point, [0.0] * main_nodes * (longest_label - iso_lengths[i])
    )
    final_dataset.append(final_data_point)
    # print(final_data_point)
    i += 1

print("NOT contained subgraphs: ", balance)
print("CONTAINED: ", graphs_generated - balance)
print("MUST be: ", 1500)

final_dataset_torch = torch.tensor(np.array(final_dataset), dtype=torch.float)
torch.save(
    final_dataset_torch,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph/graph-8_subgraph-5_3000_iso_test.pt",
)

"""
Final datapoint:
- main^2 elements -> main graph adjacency matrix;
- sub^2 elements -> subgraph adjacency matrix;
- (main + sub)^2 elements -> combined adjacency of main and sub;
- remainder -> label.
"""

""" 2. Dataset for training, containing average involvement per node. """

labels_start = main_nodes**2 + sub_nodes**2 + (main_nodes + sub_nodes) ** 2

Y = final_dataset_torch[
    :,
    labels_start:,
]  # take everything after all the flattened adjecencies to be the main label

Y = Y.reshape(
    Y.shape[0], -1, main_nodes
)  # break up all the lables into groups of num_qubits (6, or 8) (for each found isomorphism)

# print("Y_final shape: ", Y.shape)
# print("Y: ", Y)

y_list = Y.tolist()
y_final = []

for y_point in y_list:
    no_iso = 0
    iso = []
    for config in range(len(y_point)):
        if float(1) not in y_point[config]:
            no_iso += 1
        else:
            iso.append(config)

    if no_iso == len(y_point):
        y_final.append([float(0)] * main_nodes)
    else:
        nodes_av_label = [0] * main_nodes

        for config in iso:
            for i in range(main_nodes):
                nodes_av_label[i] = nodes_av_label[i] + y_point[config][i]
        y_final_point = [x / len(iso) for x in nodes_av_label]
        y_final.append(y_final_point)

print(len(y_final))
# print(y_final)

new_dataset = []
j = 0
for point in final_dataset:
    point = point[: -(main_nodes * longest_label)]  # remove all the old labels
    for elem in y_final[j]:
        point = np.append(point, elem)  # add each average to the datapoint
    new_dataset.append(point)
    j += 1

final_dataset_train_torch = torch.tensor(np.array(new_dataset), dtype=torch.float)


torch.save(
    final_dataset_train_torch,
    "/Users/home/Quantum_Computing/Pennylane/Graph_ML/data/subgraph/graph-8_subgraph-5_3000_iso_train.pt",
)

"""
Final datapoint:
- main^2 elements -> main graph adjacency matrix;
- sub^2 elements -> subgraph adjacency matrix;
- (main + sub)^2 elements -> combined adjacency of main and sub;
- remainder -> label (average for each of main_nodes nodes).
"""
