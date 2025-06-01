"""Contains the fit function for supervised learning to measure performance of a circuit in a specific experiment."""

import collections

import pennylane as qml
import torch
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd


def fit(circuit: qml.QNode,
        circuit_weight_shapes: dict[str, tuple],
        dataset: torch.utils.data.TensorDataset,
        samplings: int = 30,
        epochs: int = 50) -> tuple[dict[str, list[tuple[int, int, list[float], list[float]]]],
                                   dict[str, list[tuple[int, list[float], list[float]]]],
                                   list[tuple[int, int, list[float]]]]:
    
    # We shuffle the dataset
    subsampling = ShuffleSplit(n_splits=samplings, train_size=100, test_size=2900)

    prediction_history: dict[str, list[tuple[int, int, list[float], list[float], list[float]]]] = {"test": [], "train": []}
    targets: dict[str, list[tuple[int, list[float], list[float], list[float]]]] = {"test": [], "train": []}
    weight_history: list[tuple[int, int, list[float]]] = []
    init_weights = collections.OrderedDict([(arg_name, torch.rand(shape)) for arg_name, shape in circuit_weight_shapes.items()])

    for sampling, (train_indices, test_indices) in enumerate(subsampling.split(dataset)):

        x_train, y_train, z_train = dataset[train_indices]
        x_test, y_test, z_test = dataset[test_indices]

        targets["train"].append((sampling, x_train.tolist(), y_train.tolist(), z_train.tolist()))
        targets["test"].append((sampling, x_test.tolist(),  y_test.tolist(), z_test.tolist()))

        model = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        model.load_state_dict(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
        loss = torch.nn.BCEWithLogitsLoss()

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=25, shuffle=True)

        model.eval()
        with torch.no_grad():
            prediction_history["train"].append((sampling, 0, x_train.tolist(), model(x_train).tolist(), z_train.tolist()))
            prediction_history["test"].append((sampling, 0, x_test.tolist(), model(x_test).tolist(), z_test.tolist()))

        for epoch in range(epochs):
            #if(epoch % 5 == 0):
            print("Sampling: ", sampling, "; Epoch: ", epoch)
            model.train()
            for xs, ys in data_loader:
                optimizer.zero_grad()
                loss_evaluated = loss(model(xs), ys)
                #print('Loss: ', loss_evaluated.item())
                #print('Shape of the model:', model(xs).shape)
                #print('Model output: ', model(xs))
                #print('Target shape: ', ys.shape)
                #print('Target: ', ys)
                loss_evaluated.backward()
                optimizer.step()

            #weight_history.append((sampling, epoch, model.state_dict().get("weights").reshape(-1).tolist()))

            model.eval()
            with torch.no_grad():
                prediction_history["train"].append((sampling, epoch + 1, x_train.tolist(), model(x_train).tolist(), z_train.tolist()))
                prediction_history["test"].append((sampling, epoch + 1, x_test.tolist(),  model(x_test).tolist(), z_test.tolist()))

    print("Finished training.")
    return prediction_history, targets, weight_history
