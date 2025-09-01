import random
import pandas as pd
import matplotlib.pyplot as plt
import wandb


def baseline_random_performance(nodes_main, nodes_sub, samples, epochs, inputs):
    rows = []
    for sample in range(samples):
        for epoch in range(epochs + 1):
            for input in range(inputs):
                random_prediction = [1.0] * nodes_sub + [0.0] * (nodes_main - nodes_sub)
                random.shuffle(random_prediction)
                rows.append(
                    {
                        "sampling": sample,
                        "epoch": epoch,
                        "prediction": random_prediction,
                    }
                )

    predictions = pd.DataFrame(rows)
    return predictions


def extract_prediction_accuracy_per_qubit_isomorph(
    prediction_file_name,
    target_file_name,
    num_samples,
    num_epochs,
    num_inputs,
    num_nodes,
):

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

    # joined_table["prediction"] = joined_table["prediction"].apply(
    #    lambda list: [1.0 if x > 0 else 0.0 for x in list]
    # )

    non_iso = [[0.0] * num_nodes for _ in range(len(joined_table["target"][0]))]

    joined_table["Isomorphism"] = joined_table["target"].apply(lambda x: x != non_iso)

    joined_table["correct"] = joined_table.apply(
        lambda row: (
            row["prediction"] in row["target"]
            if row["Isomorphism"]
            else row["prediction"] == [0.0] * num_nodes
        ),
        axis=1,
    )

    joined_table = joined_table.drop(["prediction", "target", "Isomorphism"], axis=1)

    # Add up all 2900 results for same sampling and same epoch
    grouped = joined_table.groupby(["sampling", "epoch"]).sum() / num_inputs
    epoch_stats = grouped.groupby("epoch").agg(["mean", "std"])
    epoch_stats = epoch_stats.reset_index()
    epoch_stats.columns = ["Epoch", "Mean accuracy", "Error"]

    # Add the current accuracy plot to the final list of graphics

    accuracy_plot_name = (
        f"output/subgraph_per_qubit/Random_baseline/Random_baseline_accuracy.png"
    )

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
    plt.title(f"Random baseline accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(accuracy_plot_name, dpi=300, bbox_inches="tight")

    return accuracy_plot_name


samples = 5
epochs = 50
inputs = 2900
nodes_main = 8
nodes_sub = 5

baseline_random_performance(nodes_main, nodes_sub, samples, epochs, inputs).to_csv(
    "output/subgraph_per_qubit/Random_baseline/Random_predictions_main-8_sub-5_samples-5_epochs-50_inputs-2900.csv",
    index=False,
)

# Data to compare:
random_prediction = f"output/subgraph_per_qubit/Random_baseline/Random_predictions_main-8_sub-4_samples-5_epochs-50_inputs-2900.csv"

target = f"output/subgraph_per_qubit/Random_baseline/Targets_main-8_sub-5_samples-5_epochs-50_inputs-2900.csv"

# Compose the WandB summary:

summary_run = wandb.init(
    project=("Subgraph search per qubit with Pennylane"),
    group="Random baselines",
    name="Random baseline accuracy main 8 sub 5",
)

accuracy = extract_prediction_accuracy_per_qubit_isomorph(
    random_prediction,
    target,
    samples,
    epochs + 1,
    inputs,
    nodes_main,
)

table = wandb.Table(columns=["Accuracy"])

acc_img = wandb.Image(accuracy)

table.add_data(acc_img)

wandb.log({"Random baseline accuracy": table})
wandb.finish()
