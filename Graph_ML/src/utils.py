"""Defines functions to create datasets and different utils."""
from __future__ import annotations

import json
import random
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pennylane as qml
import torch
import pandas as pd

def append_to_csv(data, path, columns):
    if "target" in columns:
        pd.DataFrame(data)[columns].explode("target").to_csv(path, mode="a", header=False, index=False)
    else:
        pd.DataFrame(data)[columns].explode("prediction").to_csv(path, mode="a", header=False, index=False)

def graph_connectedness(*, num_graphs: int, num_nodes: int,
                        num_edges: int | float, stop_after: int = 100_000) -> torch.utils.data.TensorDataset:
    """Create a dataset for graph connectedness classification."""
    connected_graphs: set[tuple[tuple, int]] = set()
    disconnected_graphs: set[tuple[tuple, int]] = set()

    i = 0
    while (len(connected_graphs) < num_graphs / 2 or len(disconnected_graphs) < num_graphs / 2) and i < stop_after:
        G = nx.gnm_random_graph(num_nodes, num_edges) if isinstance(num_edges, int) else nx.gnp_random_graph(num_nodes, num_edges)
        if nx.is_connected(G) and len(connected_graphs) < num_graphs / 2:
            connected_graphs.add((tuple(torch.tensor(nx.to_numpy_array(G)).reshape(-1).tolist()), 1))
        elif len(disconnected_graphs) < num_graphs / 2:
            disconnected_graphs.add((tuple(torch.tensor(nx.to_numpy_array(G)).reshape(-1).tolist()), -1))
        i += 1

    data = [*connected_graphs, *disconnected_graphs]

    if len(data) < num_graphs:
        warnings.warn(f"Only {len(data)} graphs were generated ({len(connected_graphs)} with \
                        crossing edges and {len(disconnected_graphs)} without).", stacklevel=2)

    random.shuffle(data)
    X, Y = list(zip(*data))
    return torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())

def has_crossing_edges(edgelist: list[tuple[int, int]]) -> bool:
    """Detect if a list of edges (i, j) has crossing edges."""
    for (a, b) in edgelist:
        for (c, d) in edgelist:
            if (a < c < b) and (d > b) and len({a, b, c, d}) == 4:
                return True
    return False

def crossing_edges(*, num_graphs: int, num_nodes: int, num_edges: int, stop_after: int = 100_000) -> torch.utils.data.TensorDataset:
    """Create a dataset for crossing edges classification."""
    crossing_edges: set[tuple[tuple, int]] = set()
    no_crossing_edges: set[tuple[tuple, int]] = set()

    i = 0
    while (len(crossing_edges) < num_graphs / 2 or len(no_crossing_edges) < num_graphs / 2) and i < stop_after:
        G = nx.gnm_random_graph(num_nodes, num_edges) if isinstance(num_edges, int) else nx.gnp_random_graph(num_nodes, num_edges)
        if has_crossing_edges(G.edges) and len(crossing_edges) < num_graphs / 2:
            crossing_edges.add((tuple(torch.tensor(nx.to_numpy_array(G)).reshape(-1).tolist()), 1))
        elif len(no_crossing_edges) < num_graphs / 2:
            no_crossing_edges.add((tuple(torch.tensor(nx.to_numpy_array(G)).reshape(-1).tolist()), -1))
        i += 1

    data = [*crossing_edges, *no_crossing_edges]

    if len(data) < num_graphs:
        warnings.warn(f"Only {len(data)} graphs were generated ({len(crossing_edges)} with \
                        crossing edges and {len(no_crossing_edges)} without).", stacklevel=2)
    random.shuffle(data)
    X, Y = list(zip(*data))
    return torch.utils.data.TensorDataset(torch.tensor(X).float(), torch.tensor(Y).float())

def load_patterns(file_path: str, num_nodes: int) -> torch.utils.data.TensorDataset:
    """Load dataset."""
    patterns = torch.load(file_path)
    X = patterns[:, :num_nodes**2]
    Y = patterns[:, num_nodes**2]
    return torch.utils.data.TensorDataset(X.float(), Y.float())

def load_patterns_per_qubit(file_path: str, num_nodes: int) -> torch.utils.data.TensorDataset:
    """Load dataset."""
    patterns = torch.load(file_path)
    X = patterns[:, :num_nodes**2]
    Y = patterns[:, num_nodes**2:]
    return torch.utils.data.TensorDataset(X.float(), Y.float())

def load_patterns_per_qubit_max_cut(file_path: str, num_nodes: int) -> torch.utils.data.TensorDataset:
    """Load dataset."""
    patterns = torch.load(file_path)
    X = patterns[:, :num_nodes**2]
    Y = patterns[:, (num_nodes**2+1):]
    Z = patterns[:, num_nodes**2]
    return torch.utils.data.TensorDataset(X.float(), Y.float(), Z.float())

def intersecting_lines(*, file_path: str | Path) -> torch.utils.data.TensorDataset:
    """Load the intersecting line dataset."""
    with Path.open(file_path, "r") as file:
        content = json.load(file)
    random.shuffle(content)
    X, Y = list(zip(*content))
    return torch.utils.data.TensorDataset(torch.tensor(np.array(X)).float(), torch.tensor(Y).float())

@qml.transform
def set_observables(tape, observables):
    new_tape = type(tape)(tape.operations, observables, shots=tape.shots)

    def postprocessing(results):
        return results[0]

    return [new_tape], postprocessing
