import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def extract_prediction_accuracy(prediction_file_names, target_file_names, num_samples, num_epochs, num_inputs, just_test):
    accuracies = []
    counter = 0

    for prediction_file_name, target_file_name in zip(prediction_file_names, target_file_names):
        predictions = pd.read_csv(prediction_file_name)
        targets = pd.read_csv(target_file_name)
        if just_test[counter]:
            data = pd.DataFrame(np.insert(predictions.to_numpy(), 3,
                                          np.tile(targets["target"].to_numpy().reshape(num_samples[counter], num_inputs),
                                                  1).reshape(-1), axis=1),
                                columns=["sampling", "epoch", "prediction", "target"])
        else:
            data = pd.DataFrame(np.insert(predictions.to_numpy(), 3,
                                          np.tile(targets["target"].to_numpy().reshape(num_samples[counter], num_inputs),
                                                  num_epochs).reshape(-1), axis=1),
                                columns=["sampling", "epoch", "prediction", "target"])
        data["prediction"] = np.sign(data["prediction"])
        data.loc[data["prediction"] == -1.0, "prediction"] = 0.0
        data["prediction"] = data["prediction"] == data["target"]

        # grouped = data[["sampling", "epoch", "prediction"]].groupby(["sampling", "epoch"]).mean()
        # print(grouped.head())  # See the grouped mean accuracy for a few entries

        if just_test[counter]:  # assumes also only 1 sample
            means = data[["sampling", "epoch", "prediction"]].groupby(["sampling", "epoch"]).mean().to_numpy()
            values_all_epochs = [np.nan] * (num_epochs - 1)
            values_all_epochs.append(means[0][0])
            means = np.array(values_all_epochs).reshape(num_samples[counter], num_epochs)
            accuracies.append(means)
        else:
            accuracies.append(data[["sampling", "epoch", "prediction"]].groupby(["sampling", "epoch"]). \
                              mean().to_numpy().reshape(num_samples[counter], num_epochs))

        counter = counter + 1

    return accuracies


def plot_graph(graph_name, accuracies, num_epochs):
    _, ax = plt.subplots(figsize=(7, 4))
    colors = [
        (100 / 255, 143 / 255, 255 / 255),
        (254 / 255, 97 / 255, 0 / 255),
        (128 / 255, 128 / 255, 0 / 255)
        #(255 / 255, 176 / 255, 0 / 255),
        #(255 / 255, 176 / 255, 120 / 255),
    ]

    for data, color in zip(accuracies, colors):
        y = data.mean(axis=0)
        yerr = data.std(axis=0)
        x = list(range(num_epochs))
        # Check for exactly one non-NaN value in y
        valid_indices = ~np.isnan(y)
        if np.sum(valid_indices) == 1:
            idx = np.where(valid_indices)[0][0]  # get the index of the valid y value
            ax.plot(x[idx], y[idx], marker='o', color=color)
        else:
            ax.errorbar(x=x,
                        y=y,
                        yerr=yerr,
                        fmt="-o", markersize=3, capsize=3,
                        color=color,
                        ecolor=(*color, 0.3))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average test accuracy")
    plt.legend(["IBM Kingston Hardware", "Pennylane default gradient (sample 0)", "Pennylane default gradient (average)"])
    plt.grid()
    plt.title(graph_name)
    folder, _ = os.path.split(prediction_file_names[0])
    plt.savefig(f"{folder}/{graph_name}_simulators_and_hardware_trained_on_simulator.pdf")
    plt.show()



prediction_file_names = [
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-predictions-test-7251501.csv",
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-predictions-test-7251122-s0.csv",
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-predictions-test-7251122.csv"
]
target_file_names = [
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-targets-test-7251501.csv",
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-targets-test-7251122-s0.csv",
    "/Users/danielles/Documents/Projekte/BAIQO/invariant_circuits/Pennylane_ML/Graph_ML/output/connectedness/GC-Cn_circuit-8-90-sampling_10-epochs_50-targets-test-7251122.csv"
]

num_samples = [1, 1, 10]
just_test = [True, False, False]

num_epochs = 51
num_inputs = 50

accuracies = extract_prediction_accuracy(prediction_file_names, target_file_names, num_samples, num_epochs, num_inputs, just_test)

plot_graph("8 Qubits - Graph connectedness", accuracies, num_epochs)
