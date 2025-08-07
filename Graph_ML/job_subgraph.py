"""Main file used to run experiments and save the results."""

import argparse
import math
import pathlib
import pandas as pd
import pennylane as qml
import torch

from src import circuit, performance_subgraph, performance_subgraph_per_qubit, utils


torch.manual_seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", help="Number of layers", default=1, type=int)
    parser.add_argument("-q", "--qubits", help="Number of qubits", default=3, type=int)
    parser.add_argument(
        "-sub",
        "--subsize",
        help="Size of the subgraph to be found",
        default=4,
        type=int,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Number to add to filenames to prevent overrides",
        default=3,
        type=int,
    )
    parser.add_argument(
        "-b",
        "--base",
        help="Base directory",
        default="/home/Users/Quantum_Computing/Pennylane/Graph_ML",
        type=str,
    )
    parser.add_argument(
        "-c", "--circuit", help="Select circuit to run", default="Sn_circuit"
    )
    parser.add_argument("-f", "--find", help="Find the subgraph or not", default="n")
    parser.add_argument(
        "-d",
        "--data",
        help="filename for dataset",
        default="nodes_6-graphs_3000-edges_5_6_7.pt",
        type=str,
    )
    parser.add_argument("-e", "--epochs", help="number of epochs", default=50, type=int)
    parser.add_argument(
        "-s", "--samplings", help="number of samplings", default=30, type=int
    )

    flags = parser.parse_args()

    weight_shapes: dict[str, tuple[int, ...]] = {}

    if flags.circuit == "1":
        weight_shapes = {"weights_sn": (flags.layers, 4)}  # X + Y + Sn + Sn
        circ = (
            circuit.subgraph_other_circuit_app1_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app1
        )
        approach = 1

    elif flags.circuit == "2":
        weight_shapes = {"weights_sn": (flags.layers, 4)}  # X + Y + Sn + Sn
        circ = (
            circuit.subgraph_other_circuit_app2_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app2
        )
        approach = 2

    elif flags.circuit == "3":
        weight_shapes = {
            "weights_sn": (flags.layers, 4),
            "weights_ent": (1,),
        }  # X + Y + Sn + Sn + entanglement
        circ = (
            circuit.subgraph_other_circuit_app3_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app3
        )
        approach = 3

    elif flags.circuit == "4":
        weight_shapes = {"weights_sn": (flags.layers, 6)}  # X + X + Y + Y + Sn + Sn
        circ = (
            circuit.subgraph_other_circuit_app4_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app4
        )
        approach = 4

    elif flags.circuit == "5":
        weight_shapes = {"weights_sn": (flags.layers, 6)}  # X + X + Y + Y + Sn + Sn
        circ = (
            circuit.subgraph_other_circuit_app5_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app5
        )
        approach = 5

    elif flags.circuit == "6":
        weight_shapes = {
            "weights_sn": (flags.layers, 6),
            "weights_ent": (1,),
        }  # X + X + Y + Y + Sn + Sn + entanglement
        circ = (
            circuit.subgraph_other_circuit_app6_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app6
        )
        approach = 6

    elif flags.circuit == "7":
        weight_shapes = {"weights_sn": (flags.layers, 3)}  # X + Y + Sn
        circ = (
            circuit.subgraph_other_circuit_app7_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app7
        )
        approach = 7

    elif flags.circuit == "8":
        weight_shapes = {"weights_sn": (flags.layers, 3)}  # X + Y + Sn
        circ = (
            circuit.subgraph_other_circuit_app8_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app8
        )
        approach = 8

    elif flags.circuit == "9":
        weight_shapes = {
            "weights_sn": (flags.layers, 3),
            "weights_ent": (1,),
        }  # X + Y + Sn + entanglement
        circ = (
            circuit.subgraph_other_circuit_app9_per_qubit
            if flags.find == "y"
            else circuit.subgraph_other_circuit_app9
        )
        approach = 9

    else:
        msg = f"Circuit {flags.circuit} is unkown."
        raise ValueError(msg)

    n_parameters = sum(math.prod(shape) for shape in weight_shapes.values())
    print("Number of parameters: ", n_parameters)

    dev = qml.device("default.qubit", wires=flags.qubits + flags.subsize)
    qnode = qml.QNode(circ, device=dev, interface="torch")

    base = pathlib.Path(flags.base)

    base_output = (
        pathlib.Path(
            "/Users/home/Quantum_Computing/Pennylane/Graph_ML/output/subgraph_per_qubit"
        )
        if flags.find == "y"
        else pathlib.Path(
            "/Users/home/Quantum_Computing/Pennylane/Graph_ML/output/subgraph"
        )
    )
    dataset = (
        utils.load_patterns_subgraph_isomorph_per_qubit(  # change here to isomorphism-respecting dataset splitting
            base / flags.data, flags.qubits, flags.subsize
        )
        if flags.find == "y"
        else utils.load_patterns_subgraph(
            base / flags.data, flags.qubits, flags.subsize
        )
    )

    predictions, targets, weights = (
        performance_subgraph_per_qubit.fit(
            flags.qubits,
            flags.subsize,
            flags.layers,
            n_parameters,
            flags.find,
            approach,
            qnode,
            weight_shapes,
            dataset,
            samplings=flags.samplings,
            epochs=flags.epochs,
        )
        if flags.find == "y"
        else performance_subgraph.fit(
            flags.qubits,
            flags.subsize,
            flags.layers,
            n_parameters,
            flags.find,
            approach,
            qnode,
            weight_shapes,
            dataset,
            samplings=flags.samplings,
            epochs=flags.epochs,
        )
    )

    ext = (
        f"Subgraph_per_qubit-approach-{approach}-{flags.qubits}-{flags.subsize}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"
        if flags.find == "y"
        else f"Subgraph-approach-{approach}-{flags.qubits}-{flags.subsize}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"
    )

    pd.DataFrame(targets["train"], columns=["sampling", "target"]).explode(
        "target"
    ).to_csv(base_output / (ext + f"targets-train-{flags.name}.csv"), index=False)
    # print('First file saved')
    pd.DataFrame(
        predictions["train"], columns=["sampling", "epoch", "prediction"]
    ).explode("prediction").to_csv(
        base_output / (ext + f"predictions-train-{flags.name}.csv"), index=False
    )
    # print('Second file saved')
    pd.DataFrame(targets["test"], columns=["sampling", "target"]).explode(
        "target"
    ).to_csv(base_output / (ext + f"targets-test-{flags.name}.csv"), index=False)
    # print('Third file saved')
    pd.DataFrame(
        predictions["test"], columns=["sampling", "epoch", "prediction"]
    ).explode("prediction").to_csv(
        base_output / (ext + f"predictions-test-{flags.name}.csv"), index=False
    )
    # print('Fourth file saved')

    print("Epochs flag: ", flags.epochs)

    ###### ADD EXTRA SAMPLES FOR CONNECTED ######

    # extra_data = torch.load("/Users/home/qiskit_env/Pennylane/data/graph_connectedness/nodes_8-graphs_10_edge_cases.pt")

    # extra_x = extra_data[:, :-1]  # shape: (7, 64)
    # extra_y = extra_data[:, -1]  # shape: (7,)

    # print(extra_data.size()) #(7, 65) --> 7 graphs, for each: 8x8 adjacency + 1 label

    ###### DONE WITH EXTRA SAMPLES FOR CONNECTED ######
