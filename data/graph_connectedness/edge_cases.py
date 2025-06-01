import random

import networkx as nx
import numpy as np
import torch
import matplotlib.pyplot as plt

dataset = []

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))
for i in range(7):
    for j in range(i + 1, 7):
        G.add_edge(i, j)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))
for i in range(7):
    for j in range(i + 1, 7):
        G.add_edge(i, j)

G.add_edge(7, 0)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))
for i in range(4):
    for j in range(i + 1, 4):
        G.add_edge(i, j)

for i in range(4, 8):
    for j in range(i + 1, 8):
        G.add_edge(i, j)


array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))

for i in range(7):  # Connect sequentially
    G.add_edge(i, i + 1)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))

for i in range(7):  
    G.add_edge(i, i + 1)

G.add_edge(7, 0)
G.remove_edge(2, 3)
G.remove_edge(3, 4)
G.add_edge(4, 6)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))

for i in range(1, 8):
    G.add_edge(0, i)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

########################################################
G = nx.Graph()
G.add_nodes_from(range(8))

for i in range(1, 8):
    G.add_edge(0, i)

extra_edges = [(1, 2), (3, 4), (5, 6)]  # Example extra edges
G.add_edges_from(extra_edges)

array = np.append(nx.to_numpy_array(G).flatten(), nx.is_connected(G) and nx.is_bipartite(G))
dataset.append(array)

dataset = torch.tensor(np.array(dataset))
torch.save(dataset, "/Users/home/qiskit_env/Pennylane/data/graph_connectedness/nodes_8-graphs_10_edge_cases.pt")
