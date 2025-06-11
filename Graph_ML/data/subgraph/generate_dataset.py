import random

import networkx as nx
import numpy as np
import torch

dataset = []
num_samples = 3000
qubits = 8
subgraph_size = 3

# TODO

dataset = torch.tensor(np.array(dataset))
torch.save(dataset, "/Users/home/qiskit_env/Pennylane/data/subgraph/nodes_8-graphs_3000.pt")
