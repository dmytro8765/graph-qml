"""Contains the fit function to measure performance of a circuit in a specific experiment."""

import collections

import pennylane as qml
import torch
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd

from pennylane_qiskit import qiskit_session, RemoteDevice

from .utils import append_to_csv


def fit(circuit: qml.QNode,
        circuit_weight_shapes: dict[str, tuple],
        dataset: torch.utils.data.TensorDataset,
        device: RemoteDevice,
        file_names: dict,
        samplings: int = 30,
        extra_data: torch.Tensor = None,
        epochs: int = 50,
        resume_at: int = -1,
        resume_after_batch: int = -1) -> tuple[dict[str, list[tuple[int, int, list[float]]]],
                                   dict[str, list[tuple[int, list[float]]]],
                                   list[tuple[int, int, list[float]]]]:
    
    # We shuffle the dataset
    subsampling = ShuffleSplit(n_splits=samplings, train_size=6, test_size=2994, random_state=42)

    prediction_history: dict[str, list[tuple[int, int, list[float]]]] = {"test": [], "train": []}
    targets: dict[str, list[tuple[int, list[float]]]] = {"test": [], "train": []}
    weight_history: list[tuple[int, int, list[float]]] = []
    if resume_at > -1:
        init_weights = None
    else:
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
        x_test, y_test = x_test[:2], y_test[:2]

        
        #targets["train"].append((sampling, y_train.tolist()))
        if resume_at == -1:
            append_to_csv(
                {"sampling": sampling, "target": y_train.tolist()},
                file_names["targets-train"],
                ["sampling", "target"]
            )
            #targets["test"].append((sampling, y_test.tolist()))
            append_to_csv(
                {"sampling": sampling, "target": y_test.tolist()},
                file_names["targets-test"],
                ["sampling", "target"]
            )

        model = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        if init_weights is not None:
            model.load_state_dict(init_weights)
            torch.save(model.state_dict(), file_names["weights"])
        else:
            model.load_state_dict(torch.load(file_names["weights"]))

        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

        loss = torch.nn.BCEWithLogitsLoss()

        data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*dataset[train_indices]), batch_size=2, shuffle=True, drop_last=False,
        )

        if resume_at == -1:
            model.eval()
            with torch.no_grad():
                with qiskit_session(device, max_time=345600) as session:
                    pred_train_before = model(x_train).tolist()
                #prediction_history["train"].append((sampling, 0, pred_train_before))
                append_to_csv(
                    {"sampling": sampling, "epoch": 0, "prediction": pred_train_before},
                    file_names["predictions-train"],
                    ["sampling", "epoch", "prediction"]
                )
                with qiskit_session(device, max_time=345600) as session:
                    pred_test_before = model(x_test).tolist()
                #prediction_history["test"].append((sampling, 0, pred_test_before))
                append_to_csv(
                    {"sampling": sampling, "epoch": 0, "prediction": pred_test_before},
                    file_names["predictions-test"],
                    ["sampling", "epoch", "prediction"]
                )
            start_epoch = 0
        else:
            start_epoch = resume_at

        for epoch in range(start_epoch, epochs):
            #if(epoch % 10 == 0):
            print("Sampling: ", sampling, "; Epoch: ", epoch)
            model.train()
            #pred_epoch = []
            batch_done = 0
            for xs, ys in data_loader:
                if epoch == start_epoch and -1 < resume_after_batch and batch_done < resume_after_batch:
                    batch_done += 1
                    print(f"Skip batch {batch_done} of starting epoch")
                else:
                    optimizer.zero_grad()
                    print("    Starting forward pass.")
                    with qiskit_session(device, max_time=345600) as session:
                        pred_batch = model(xs)
                    append_to_csv(
                        {"sampling": sampling, "epoch": epoch + 1, "prediction": pred_batch.tolist()},
                        file_names["predictions-train"],
                        ["sampling", "epoch", "prediction"]
                    )
                    print("    Forward pass done.")
                    #pred_epoch.extend(pred_batch.tolist())
                    loss_evaluated = loss(pred_batch, ys)
                    print("    Loss calculated.")
                    with qiskit_session(device, max_time=345600) as session:
                        loss_evaluated.backward()
                    print("    Backward pass done.")
                    optimizer.step()
                    print("    Optimizer step done.")
                    torch.save(model.state_dict(), file_names["weights"])
                    print("    Model saved.")
                    print("Another batch (25 data points) done.\n\n\n")

            #weight_history.append((sampling, epoch, model.state_dict().get("weights").reshape(-1).tolist()))
            #prediction_history["train"].append((sampling, epoch + 1, pred_epoch))

            model.eval()
            with torch.no_grad():

                #logits_train = model(x_train)
                #preds_train = (torch.sigmoid(logits_train))
                #y_train_int = y_train.int()

                #prediction_history["train"].append((sampling, epoch + 1, model(x_train).tolist()))
                #prediction_history["test"].append((sampling, epoch + 1, model(x_test).tolist()))
                print("Start test")
                with qiskit_session(device, max_time=345600) as session:
                    test_results = model(x_test)
                append_to_csv(
                    {"sampling": sampling, "epoch": epoch + 1, "prediction": test_results.tolist()},
                    file_names["predictions-test"],
                    ["sampling", "epoch", "prediction"]
                )
                print("    Test done.\n\n\n")


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

        if extra_data is not None:
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
