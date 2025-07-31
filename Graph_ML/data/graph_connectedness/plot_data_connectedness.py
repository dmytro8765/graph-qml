import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import ShuffleSplit

# Load the saved dataset
cwd = os.getcwd()
path = f"{cwd}/nodes_8-graphs_3000.pt"
dataset = torch.load(path)
subsampling = ShuffleSplit(n_splits=1, train_size=100, test_size=2900, random_state=42)

# Initialize holders for graphs
first_connected = None
first_disconnected = None

for i, data in enumerate(dataset):
    adj_flat = data[:-1].numpy()
    label = int(data[-1].item())
    adj_matrix = adj_flat.reshape((8, 8))

    # Ensure it's symmetric (should be, but just to be safe)
    adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T

    G = nx.from_numpy_array(adj_matrix)

    if label == 1 and first_connected is None:
        first_connected = G
        print(f"Found first connected graph at index {i}")
    elif label == 0 and first_disconnected is None:
        first_disconnected = G
        print(f"Found first disconnected graph at index {i}")

    if first_connected and first_disconnected:
        break

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
nx.draw(first_connected, with_labels=True, ax=axs[0], node_color='lightblue')
axs[0].set_title("Connected Graph")

nx.draw(first_disconnected, with_labels=True, ax=axs[1], node_color='lightcoral')
axs[1].set_title("Disconnected Graph")

#plt.tight_layout()
plt.savefig(f"{cwd}/example_connectedness_data.pdf")
plt.show()
