"""
Contains the fit function for supervised learning to measure performance of
a circuit in a specific experiment.

Prediction: per qubit.
"""

import collections

import pennylane as qml
import torch
from sklearn.model_selection import ShuffleSplit
import numpy as np


def fit(
    circuit: qml.QNode,
    circuit_weight_shapes: dict[str, tuple],
    dataset: torch.utils.data.TensorDataset,
    samplings: int = 30,
    # extra_data: torch.Tensor = None,
    epochs: int = 50,
) -> tuple[
    dict[str, list[tuple[int, int, list[float]]]],
    dict[str, list[tuple[int, list[float]]]],
    list[tuple[int, int, list[float]]],
]:

    # We shuffle the dataset
    subsampling = ShuffleSplit(n_splits=samplings, train_size=100, test_size=2900)

    prediction_history: dict[str, list[tuple[int, int, list[float]]]] = {
        "test": [],
        "train": [],
    }
    targets: dict[str, list[tuple[int, list[float]]]] = {"test": [], "train": []}
    weight_history: list[tuple[int, int, list[float]]] = []
    init_weights = collections.OrderedDict(
        [
            (arg_name, torch.rand(shape))
            for arg_name, shape in circuit_weight_shapes.items()
        ]
    )

    for sampling, (train_indices, test_indices) in enumerate(
        subsampling.split(dataset)
    ):

        x_train, y_train = dataset[train_indices]
        x_test, y_test = dataset[test_indices]

        targets["train"].append((sampling, y_train.tolist()))
        targets["test"].append((sampling, y_test.tolist()))

        model = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        model.load_state_dict(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

        loss = torch.nn.BCEWithLogitsLoss()

        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=25, shuffle=True
        )

        # data_loader = torch.utils.data.DataLoader(
        #    torch.utils.data.TensorDataset(*dataset[train_indices]), batch_size=25, shuffle=True, drop_last=False,
        # )

        model.eval()
        with torch.no_grad():
            prediction_history["train"].append((sampling, 0, model(x_train).tolist()))
            prediction_history["test"].append((sampling, 0, model(x_test).tolist()))

        for epoch in range(epochs):
            # if(epoch % 5 == 0):
            print("Sampling: ", sampling, "; Epoch: ", epoch)
            model.train()
            for xs, ys in data_loader:
                optimizer.zero_grad()
                loss_evaluated = loss(model(xs), ys)
                # print('Loss: ', loss_evaluated.item())
                # print('Shape of the model:', model(xs).shape)
                # print('Model output: ', model(xs))
                # print('Target shape: ', ys.shape)
                # print('Target: ', ys)
                loss_evaluated.backward()
                optimizer.step()

            # weight_history.append((sampling, epoch, model.state_dict().get("weights").reshape(-1).tolist()))

            model.eval()
            with torch.no_grad():
                """train_preds = model(x_train)
                test_preds = model(x_test)

                # Step 4: Sanity check for first few samples
                print(f"\nSample predictions vs targets (Epoch {epoch + 1}):")
                for i in range(3):  # Show 3 examples
                    print(f"Train Pred {i}: {torch.sigmoid(train_preds[i])} → Target: {y_train[i]}")
                    print(f"Test  Pred {i}: {torch.sigmoid(test_preds[i])} → Target: {y_test[i]}")
                print()"""

                prediction_history["train"].append(
                    (sampling, epoch + 1, model(x_train).tolist())
                )
                prediction_history["test"].append(
                    (sampling, epoch + 1, model(x_test).tolist())
                )

    return prediction_history, targets, weight_history
