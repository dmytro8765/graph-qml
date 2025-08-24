"""
Train a parameterized quantum neural network circuit,
using Pennylane and PyTorch on subsampled dataset splits.

The function performs multiple random train/test splits (controlled by `samplings`),
initializing the model weights and optimizer for each split.

It preprocesses target labels by averaging over isomorphic configurations, then trains the model
for a given number of epochs while logging loss and predictions using Weights & Biases (wandb).

Predict a value for each node of the graph separately.

Returns dictionaries containing prediction histories and target values for both train and test sets,
as well as weight history for further analysis or debugging.
"""

import collections
import pennylane as qml
import torch
from sklearn.model_selection import ShuffleSplit
import wandb
import numpy as np


def fit(
    main_size: int,
    sub_size: int,
    layers: int,
    parameters: int,
    find: str,
    approach: int,
    circuit: qml.QNode,
    circuit_weight_shapes: dict[str, tuple],
    dataset_train: torch.utils.data.TensorDataset,
    dataset_test: torch.utils.data.TensorDataset,
    samplings: int = 30,
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

    # print(init_weights)
    # print(circuit_weight_shapes)

    group_name = f"Circuit_isomorph_{approach}"

    for sampling, (train_indices, test_indices) in enumerate(
        subsampling.split(dataset_train)
    ):

        # Initialize a new W&B run per sample with grouping by circuit
        run_name = f"Circuit_{approach}-sample_{sampling+1}"
        print(find)
        wandb_run = wandb.init(
            project=(
                "Subgraph search per qubit with Pennylane"
                if find == "y"
                else "Binary subgraph search with Pennylane"
            ),
            group=group_name,
            name=run_name,
            config={
                "Layers": layers,
                "Parameters": parameters,
                "Main": main_size,
                "Sub": sub_size,
                "Circuit": approach,
            },
        )

        # print(dataset[train_indices])

        x_main_train, x_sub_train, big_input_train, y_train = dataset_train[
            train_indices
        ]
        x_main_test, x_sub_test, big_input_test, y_test = dataset_test[test_indices]

        # print(y_train.tolist())

        targets["train"].append((sampling, y_train.tolist()))
        targets["test"].append((sampling, y_test.tolist()))

        model = qml.qnn.TorchLayer(circuit, circuit_weight_shapes)
        model.load_state_dict(init_weights)

        """# ---------------- Take a single datapoint (first one from training set) ----------------
        sample_input = big_input_train[0].detach().numpy()
        sample_weights = init_weights["weights_sn"].detach().numpy()

        # Draw the circuit with the sample input + init weights
        drawer = qml.draw(circuit)
        print(drawer(sample_input, sample_weights))
        # ---------------- End of drawing ----------------"""

        optimizer = torch.optim.SGD(model.parameters(), lr=0.08)
        loss = torch.nn.BCEWithLogitsLoss()

        train_dataset = torch.utils.data.TensorDataset(
            x_main_train,
            x_sub_train,
            big_input_train,
            torch.tensor(np.array(y_train), dtype=torch.float),
        )
        data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=25, shuffle=True
        )

        model.eval()
        with torch.no_grad():
            prediction_history["train"].append(
                (sampling, 0, model(big_input_train).tolist())
            )
            prediction_history["test"].append(
                (sampling, 0, model(big_input_test).tolist())
            )

        for epoch in range(epochs):
            # if(epoch % 5 == 0):
            print("Sampling: ", sampling, "; Epoch: ", epoch)

            model.train()
            for xs_main, xs_sub, big_input, ys in data_loader:
                optimizer.zero_grad()
                loss_evaluated = loss(model(big_input), ys)
                loss_evaluated.backward()
                optimizer.step()

            # Log loss and epoch
            wandb_run.log({"epoch": epoch + 1, "loss": loss_evaluated.item()})

            model.eval()
            with torch.no_grad():
                prediction_history["train"].append(
                    (sampling, epoch + 1, model(big_input_train).tolist())
                )
                prediction_history["test"].append(
                    (sampling, epoch + 1, model(big_input_test).tolist())
                )

        # Finish wandb run cleanly
        wandb_run.finish()

    return prediction_history, targets, weight_history
