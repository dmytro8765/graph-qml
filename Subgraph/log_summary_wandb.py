"""
Script for evaluating and visualizing prediction accuracy from quantum circuit experiments.

Generates accuracy plots for different circuit configurations,
with support for standard, per-qubit, and isomorphism-aware evaluation modes.

Also logs results to Weights & Biases (WandB) for experiment tracking and comparison.

Intended to be run after model training to assess performance across various experimental setups.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import argparse

# params_15 = [15, 16, 16, 17, 12, 12, 13, 15, 15, 16]
params_30 = [30, 28, 28, 29, 30, 30, 31, 30, 30, 31]
params_60 = [60, 60, 60, 61, 60, 60, 61, 60, 60, 61]
params_90 = [90, 88, 88, 89, 90, 90, 91, 90, 90, 91]
params_120 = [120, 20, 120, 121, 120, 120, 121, 120, 120, 121]

default_params = params_60

param_dict = {
    # 15: params_15,
    30: params_30,
    60: params_60,
    90: params_90,
    120: params_120,
}


def extract_prediction_accuracy(
    prediction_file_names,
    target_file_names,
    num_samples,
    num_epochs,
    num_inputs,
    exec_circuits,
    ext_acc,
):

    accuracies = []

    for prediction_file_name, target_file_name in zip(
        prediction_file_names, target_file_names
    ):
        predictions = pd.read_csv(prediction_file_name)
        targets = pd.read_csv(target_file_name)

        data = pd.DataFrame(
            np.insert(
                predictions.to_numpy(),
                3,
                np.tile(
                    targets["target"].to_numpy().reshape(num_samples, num_inputs),
                    num_epochs,
                ).reshape(-1),
                axis=1,
            ),
            columns=["sampling", "epoch", "prediction", "target"],
        )

        data["prediction"] = np.sign(data["prediction"])
        data.loc[data["prediction"] == -1.0, "prediction"] = 0.0
        data["prediction"] = data["prediction"] == data["target"]

        accuracies.append(
            data[["sampling", "epoch", "prediction"]]
            .groupby(["sampling", "epoch"])
            .mean()
            .to_numpy()
            .reshape(num_samples, num_epochs)
        )

        combined_df = pd.DataFrame({"Epoch": list(range(num_epochs))})

        # First add all means
        for i, data in enumerate(accuracies):
            model_id = f"Model{i+1}"
            means = data.mean(axis=0)
            combined_df[f"{model_id}_Mean"] = data.mean(axis=0)

        for i, data in enumerate(accuracies):
            model_id = f"Model{i+1}"
            errors = data.std(axis=0)
            errors = data.std(axis=0)
            combined_df[f"{model_id}_Error"] = errors

        """
        # Check the final table for outliers before exporting
        outliers = combined_df[combined_df.filter(like='_Mean').gt(1).any(axis=1)]
        if not outliers.empty:
            print("⚠️ Outliers in mean values detected:")
            print(outliers)
        """

        combined_df = combined_df.rename(
            columns={
                "Model1_Mean": "Sn_Mean",
                "Model1_Error": "Sn_Error",
                "Model2_Mean": "entanglement_Mean",
                "Model2_Error": "entanglement_Error",
                "Model3_Mean": "Energy_Mean",
                "Model3_Error": "Energy_Error",
                "Model4_Mean": "free_parameters_Mean",
                "Model4_Error": "free_parameters_Error",
            }
        )

    accuracies_plots = []
    for i, circuit_data in enumerate(accuracies):
        j = exec_circuits[i]
        filename = ext_acc + f"Circuit_{j}"

        fig, ax = plt.subplots(figsize=(7, 4))
        mean_acc = circuit_data.mean(axis=0)
        std_acc = circuit_data.std(axis=0)
        ax.errorbar(
            x=list(range(num_epochs)),
            y=mean_acc,
            yerr=std_acc,
            fmt="-o",
            markersize=3,
            capsize=3,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average test accuracy")
        ax.set_title(f"Circuit {j}")
        ax.grid(True)

        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        accuracies_plots.append(filename)

    return accuracies_plots


def extract_prediction_accuracy_node(
    prediction_file_names,
    target_file_names,
    num_samples,
    num_epochs,
    num_inputs,
    num_nodes,
    exec_circuits,
    ext_acc,
):

    accuracies = []
    accuracies_plots = []
    k = 0

    for prediction_file_name, target_file_name in zip(
        prediction_file_names, target_file_names
    ):
        predictions = pd.read_csv(prediction_file_name)
        targets = pd.read_csv(target_file_name)

        targets_smaller = targets.drop(columns=["sampling"])
        chunks = [
            targets_smaller.iloc[i * num_inputs : (i + 1) * num_inputs]
            for i in range(num_samples)
        ]
        repeated_chunks = []
        for chunk in chunks:
            repeated_chunks.append(pd.concat([chunk] * num_epochs, ignore_index=True))
        targets_repeated = pd.concat(repeated_chunks, ignore_index=True)

        joined_table = predictions.join(targets_repeated)
        joined_table["prediction"] = joined_table["prediction"].apply(
            lambda x: x if isinstance(x, list) else eval(x)
        )
        joined_table["target"] = joined_table["target"].apply(
            lambda x: x if isinstance(x, list) else eval(x)
        )

        pred_split = pd.DataFrame(
            joined_table["prediction"].to_list(),
            columns=[f"prediction_{i+1}" for i in range(num_nodes)],
        )
        joined_table = pd.concat(
            [joined_table.drop(columns=["prediction"]), pred_split], axis=1
        )

        target_split = pd.DataFrame(
            joined_table["target"].to_list(),
            columns=[f"target_{i+1}" for i in range(num_nodes)],
        )
        joined_table = pd.concat(
            [joined_table.drop(columns=["target"]), target_split], axis=1
        )
        joined_table.to_csv("JOINED_TABLE_TEST.csv")

        node_accuracies = []

        for i in range(1, num_nodes + 1):
            pred_col = f"prediction_{i}"
            target_col = f"target_{i}"
            joined_table[pred_col] = np.sign(joined_table[pred_col])
            joined_table.loc[joined_table[pred_col] == -1.0, pred_col] = 0.0
            joined_table["correct"] = joined_table[pred_col] == joined_table[target_col]
            grouped = (
                joined_table[["sampling", "epoch", "correct"]]
                .groupby(["sampling", "epoch"])
                .mean()
            )
            accuracy_matrix = grouped.to_numpy().reshape(num_samples, num_epochs)
            node_accuracies.append(accuracy_matrix)

        accuracies = np.stack(node_accuracies, axis=0)
        combined_df = pd.DataFrame({"Epoch": list(range(num_epochs))})

        for i, data in enumerate(accuracies):
            combined_df[f"Node{i+1}_Mean"] = data.mean(axis=0)
            combined_df[f"Node{i+1}_Error"] = data.std(axis=0)

        node_cols = [f"Node{i}_Mean" for i in range(1, num_nodes + 1)]
        combined_df["Node_Avg"] = combined_df[node_cols].mean(axis=1)
        combined_df.drop(columns=node_cols, inplace=True)

        node_cols_error = [f"Node{i}_Error" for i in range(1, num_nodes + 1)]
        combined_df["Node_Avg_Error"] = combined_df[node_cols_error].mean(axis=1)
        combined_df.drop(columns=node_cols_error, inplace=True)

        # Plot accuracies per circuit

        filename = ext_acc + f"Circuit_{exec_circuits[k]}"
        fig, ax = plt.subplots(figsize=(12, 8))
        color = "green"

        epochs = list(range(num_epochs))
        fat_epochs = list(range(0, num_epochs, 5))

        means = combined_df["Node_Avg"]
        stds = combined_df["Node_Avg_Error"]

        ax.plot(epochs, means, "-o", markersize=3, label="Average", color=color)
        ax.errorbar(
            epochs, means, yerr=stds, fmt="none", capsize=3, alpha=0.3, color=color
        )
        ax.plot(fat_epochs, means.iloc[::5], "o", markersize=6, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average test accuracy")
        ax.set_title(f"Circuit {exec_circuits[k]}")
        ax.legend(title="Average Accuracy")
        ax.grid(True)

        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        accuracies_plots.append(filename)

        k += 1

    return accuracies_plots


def extract_prediction_accuracy_node_isomorph(
    prediction_file_names,
    target_file_names,
    num_samples,
    num_epochs,
    num_inputs,
    num_nodes,
    exec_circuits,
    ext_acc,
):
    accuracies_plots = []
    k = 0

    for prediction_file_name, target_file_name in zip(
        prediction_file_names, target_file_names
    ):
        # print("Summarizing circuit ", k)
        predictions = pd.read_csv(prediction_file_name)
        targets = pd.read_csv(target_file_name)

        # Now: one column of targets, each row contains all configurations of found iso + remaining zeroes
        targets_smaller = targets.drop(columns=["sampling"])

        # Turn string rows into (nested) lists
        targets_smaller["target"] = targets_smaller["target"].apply(
            lambda x: x if isinstance(x, list) else eval(x)
        )

        # Split into chunks for each sampling
        chunks = [
            targets_smaller.iloc[i * num_inputs : (i + 1) * num_inputs]
            for i in range(num_samples)
        ]

        repeated_chunks = []
        for chunk in chunks:
            repeated_chunks.append(
                pd.concat([chunk] * num_epochs, ignore_index=True)
            )  # repeat each chunk for each epoch
        targets_repeated = pd.concat(repeated_chunks, ignore_index=True)

        # Add the targets to predictions, as a new column
        joined_table = predictions.join(targets_repeated)

        """
        Columns at this point: 
        sampling, epoch, prediction (list of 6 measurements), target (list of configurations).
        """

        # Turn string rows into lists
        joined_table["prediction"] = joined_table["prediction"].apply(
            lambda x: x if isinstance(x, list) else eval(x)
        )

        joined_table["prediction"] = joined_table["prediction"].apply(
            lambda list: [1.0 if x > 0 else 0.0 for x in list]
        )

        non_iso = [[0.0] * 6 for _ in range(len(joined_table["target"][0]))]

        joined_table["Isomorphism"] = joined_table["target"].apply(
            lambda x: x != non_iso
        )

        joined_table["correct"] = joined_table.apply(
            lambda row: (
                row["prediction"] in row["target"]
                if row["Isomorphism"]
                else row["prediction"] == [0.0] * num_nodes
            ),
            axis=1,
        )

        joined_table = joined_table.drop(
            ["prediction", "target", "Isomorphism"], axis=1
        )

        # Add up all 2900 result for same sampling and same epoch
        grouped = joined_table.groupby(["sampling", "epoch"]).sum() / num_inputs

        epoch_stats = grouped.groupby("epoch").agg(["mean", "std"])

        epoch_stats = epoch_stats.reset_index()
        epoch_stats.columns = ["Epoch", "Mean accuracy", "Error"]

        # Add the current accuracy plot to the final list of graphics

        filename = ext_acc + f"Isomorph_circ_{exec_circuits[k]}"

        plt.plot(
            epoch_stats["Epoch"],
            epoch_stats["Mean accuracy"],
            linestyle="-",  # line only
            color="blue",
            label="Mean over epochs",
        )

        every_5 = epoch_stats[epoch_stats["Epoch"] % 5 == 0]

        plt.errorbar(
            every_5["Epoch"],
            every_5["Mean accuracy"],
            yerr=every_5["Error"],
            fmt="o",  # markers only
            capsize=5,
        )

        plt.xlabel("Epoch")
        plt.ylabel("Mean accuracy over samplings")
        plt.title(f"Isomorph-respecting accuracy for circuit {k+1}")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches="tight")

        accuracies_plots.append(filename)

        plt.clf()  # clear the plot

        k += 1

    return accuracies_plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--qubits", help="Number of qubits", default=3, type=int)
    parser.add_argument(
        "-sub",
        "--subsize",
        help="Size of the subgraph to be found",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-c",
        "--circuits",
        nargs="+",
        help="List of executed circuits",
        default=[0],
        type=int,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Number to add to filenames to prevent overrides",
        default=3,
        type=int,
    )
    parser.add_argument("-f", "--find", help="Find the subgraph or not", default="n")
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=50, type=int)
    parser.add_argument(
        "-s", "--samplings", help="Number of samplings", default=30, type=int
    )
    parser.add_argument(
        "-i", "--inputs", help="Number of test inputs", default=2900, type=int
    )
    parser.add_argument(
        "-pt", "--target", help="Parameter target", default=15, type=int
    )

    flags = parser.parse_args()

# Location to store accuracies
ext_accuracies = (
    f"Subgraph/node/output/accuracies/"
    if flags.find == "y"
    else f"Subgraph/graph/output/accuracies/"
)

# location to store output tables
ext_output = (
    f"/Users/home/Quantum_Computing/Pennylane/Subgraph/node/output/"
    if flags.find == "y"
    else f"/Users/home/Quantum_Computing/Pennylane/Subgraph/graph/output/"
)

parameters = param_dict.get(flags.target, default_params)

target_file_names = []
prediction_file_names = []
summary_name = ""

for i in range(len(flags.circuits)):
    j = flags.circuits[i]
    num_params = parameters[j]

    target_file_names.append(
        ext_output
        + (
            "Results_approach_" + str(j) + "/Subgraph_node-approach-"
            if flags.find == "y"
            else "Results_approach_" + str(j) + "/Subgraph_graph-approach-"
        )
        + str(j)
        + "-"
        + str(flags.qubits)
        + "-"
        + str(flags.subsize)
        + "-"
        + str(num_params)
        + "-sampling_"
        + str(flags.samplings)
        + "-epochs_"
        + str(flags.epochs)
        + "-targets-test-0.csv"
    )
    prediction_file_names.append(
        ext_output
        + (
            "Results_approach_" + str(j) + "/Subgraph_node-approach-"
            if flags.find == "y"
            else "Results_approach_" + str(j) + "/Subgraph_graph-approach-"
        )
        + str(j)
        + "-"
        + str(flags.qubits)
        + "-"
        + str(flags.subsize)
        + "-"
        + str(num_params)
        + "-sampling_"
        + str(flags.samplings)
        + "-epochs_"
        + str(flags.epochs)
        + "-predictions-test-0.csv"
    )

    summary_name += str(j) + "_"

summary_run = wandb.init(
    project=(
        "Subgraph search per node with Pennylane"
        if flags.find == "y"
        else "Binary subgraph search with Pennylane"
    ),
    group="Summaries",
    name="Summary of circuit(s): " + summary_name + str(flags.target),
)

if flags.find == "y":
    accuracies = extract_prediction_accuracy_node_isomorph(
        prediction_file_names,
        target_file_names,
        flags.samplings,
        flags.epochs + 1,
        flags.inputs,
        flags.qubits,
        flags.circuits,
        ext_accuracies,
    )
else:
    accuracies = extract_prediction_accuracy(
        prediction_file_names,
        target_file_names,
        flags.samplings,
        flags.epochs + 1,
        flags.inputs,
        flags.circuits,
        ext_accuracies,
    )

table = wandb.Table(
    columns=["Circuit number", "Circuit diagram", "Parameters", "Accuracies"]
)

# Location of all circuit diagrams
ext_diag = (
    f"Subgraph/node/output/circuit_diagrams/circuit_"
    if flags.find == "y"
    else f"Subgraph/graph/output/circuit_diagrams/circuit_"
)

circuit_diagrams = []
for i in range(10):
    circuit_diagrams.append(ext_diag + str(i))

# print(circuit_diagrams)

for i in range(len(flags.circuits)):
    j = flags.circuits[i]
    diagram_img = wandb.Image(
        f"{circuit_diagrams[j]}.png"
    )  # j-1, not i, since 1, 3, 5 --> i will still give 0, 1, 2
    acc_img = wandb.Image(f"{accuracies[i]}.png")

    table.add_data(j, diagram_img, parameters[i], acc_img)

wandb.log({"Circuit summary": table})
wandb.finish()
