"""Contains the fit function to measure performance of a circuit in a specific experiment."""

import collections

import pennylane as qml
import torch
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt


def fit(circuit: qml.QNode,
        circuit_weight_shapes: dict[str, tuple],
        dataset: torch.utils.data.TensorDataset,
        samplings: int = 30,
        extra_data: torch.Tensor = None,
        epochs: int = 50) -> tuple[dict[str, list[tuple[int, int, list[float]]]],
                                   dict[str, list[tuple[int, list[float]]]],
                                   list[tuple[int, int, list[float]]]]:
    
    # We shuffle the dataset
    subsampling = ShuffleSplit(n_splits=samplings, train_size=10, test_size=2990)

    prediction_history: dict[str, list[tuple[int, int, list[float]]]] = {"test": [], "train": []}
    targets: dict[str, list[tuple[int, list[float]]]] = {"test": [], "train": []}
    weight_history: list[tuple[int, int, list[float]]] = []
    init_weights = collections.OrderedDict([(arg_name, torch.rand(shape)) for arg_name, shape in circuit_weight_shapes.items()])

    ######################## Block for Edge-Cases ########################
    extra_predictions_0 = []
    extra_predictions_1 = []
    extra_predictions_2 = []
    extra_predictions_3 = []
    extra_predictions_4 = []
    extra_predictions_5 = []
    extra_predictions_6 = []
    ######################## Block for Edge-Cases ########################

    for sampling, (train_indices, test_indices) in enumerate(subsampling.split(dataset)):

        x_train, y_train = dataset[train_indices]
        x_test, y_test = dataset[test_indices]

        
        targets["train"].append((sampling, y_train.tolist()))
        targets["test"].append((sampling, y_test.tolist()))

        model = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        model.load_state_dict(init_weights)

        # ====== DRAW THE CIRCUIT BEFORE TRAINING ======

        # Extract one sample input from your dataset
        sample_x = dataset.tensors[0][0]   # shape: (input_dim,)
        sample_weights = init_weights["weights"]

        # Matplotlib circuit diagram
        qml.draw_mpl(circuit)(sample_x, sample_weights)
        plt.show()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

        loss = torch.nn.BCEWithLogitsLoss()

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*dataset[train_indices]), batch_size=25, shuffle=True, drop_last=False,
        )

        model.eval()
        with torch.no_grad():
            prediction_history["train"].append((sampling, 0, model(x_train).tolist()))
            prediction_history["test"].append((sampling, 0, model(x_test).tolist()))

        for epoch in range(epochs):
            if(epoch % 10 == 0):
                print("Sampling: ", sampling, "; Epoch: ", epoch)
            model.train()
            for xs, ys in data_loader:
                optimizer.zero_grad()
                loss_evaluated = loss(model(xs), ys)
                loss_evaluated.backward()
                optimizer.step()

            #weight_history.append((sampling, epoch, model.state_dict().get("weights").reshape(-1).tolist()))

            model.eval()
            with torch.no_grad():

                logits_train = model(x_train)
                preds_train = (torch.sigmoid(logits_train))
                y_train_int = y_train.int()

                prediction_history["train"].append((sampling, epoch + 1, model(x_train).tolist()))
                prediction_history["test"].append((sampling, epoch + 1, model(x_test).tolist()))

                '''print(f"\nSampling {sampling} | Epoch {epoch + 1}")
                print("TRAINING PREDICTIONS VS LABELS (per graph):")
                for idx in range(len(preds_train)):
                    pred = preds_train[idx].tolist()
                    label = y_train_int[idx].tolist()
                    print(f"Graph {idx:3d} | Pred: {pred} | Label: {label}")'''

            if extra_data is not None and epoch == epochs - 1:
                extra_x, extra_y = extra_data

                with torch.no_grad():
                    model.eval()
                    extra_pred = model(extra_x).tolist()

                print('Predictions for this sampling:', extra_pred)
                extra_predictions_0.append(extra_pred[0])
                extra_predictions_1.append(extra_pred[1])
                extra_predictions_2.append(extra_pred[2])
                extra_predictions_3.append(extra_pred[3])
                extra_predictions_4.append(extra_pred[4])
                extra_predictions_5.append(extra_pred[5])
                extra_predictions_6.append(extra_pred[6])

        final_extra_predictions = []
        final_extra_predictions.append(np.mean(extra_predictions_0))
        final_extra_predictions.append(np.mean(extra_predictions_1))
        final_extra_predictions.append(np.mean(extra_predictions_2))
        final_extra_predictions.append(np.mean(extra_predictions_3))
        final_extra_predictions.append(np.mean(extra_predictions_4))
        final_extra_predictions.append(np.mean(extra_predictions_5))
        final_extra_predictions.append(np.mean(extra_predictions_6))

        list_names = ['Fully connected + 1', 'Tethered 1', 'Clean split', 'Minimum graph', 'Minimum disconnect', 'Single tree', 'Deeper tree']

        if sampling == samplings - 1:
            for i, (pred, target) in enumerate(zip(final_extra_predictions, extra_y.tolist())):
                print(f"{list_names[i]}: Mean prediction = {pred:.4f}, Target = {target}")
                #print(f"Test {i+1}: Mean prediction = {(np.sign(pred)+1)/2}, Target = {target}")'''

    return prediction_history, targets, weight_history
