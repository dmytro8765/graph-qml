"""Main file used to run experiments and save the results."""

import argparse
import math
import pathlib

import pandas as pd
import pennylane as qml
import torch

from src import circuit, performance, utils, performance_supervised, performance_supervised_max_cut


torch.manual_seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", help="Number of layers", default=1, type=int)
    parser.add_argument("-rl", "--rotationlayers", help="Number of layers with individual rotations", default=0, type=int)
    parser.add_argument("-q", "--qubits", help="Number of qubits", default=3, type=int)
    parser.add_argument("-n", "--name", help="Number to add to filenames to prevent overrides", default=3, type=int)
    parser.add_argument("-b", "--base", help="Base directory", default="/home/zombor/ICML/SimulationsQMLSymmetries", type=str)
    parser.add_argument("-c", "--circuit", help="Select circuit to run", default="Sn_circuit")
    parser.add_argument("-t", "--task", help = "Select task to be trained for", default = "Connectedness")
    parser.add_argument("-d", "--data", help="filename for dataset", default="nodes_6-graphs_3000-edges_5_6_7.pt", type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs", default=50, type=int)
    parser.add_argument("-s", "--samplings", help="number of samplings", default=30, type=int)

    flags = parser.parse_args()

    weight_shapes: dict[str, tuple[int, ...]] = {}

    if flags.circuit == "Sn_circuit":
        weight_shapes = {"weights": (flags.layers, 3)}
        if flags.task in ["Connectedness", "Bipartiteness", "Connected_plus_Bipartite", "Hamiltonian"]:
            circ = circuit.Sn_circuit
        else: 
            circ = circuit.Sn_circuit_per_qubit

    elif flags.circuit == "entanglement_circuit":
        weight_shapes = {"weights": (flags.layers, flags.qubits)}
        if flags.task in ["Connectedness", "Bipartiteness", "Connected_plus_Bipartite", "Hamiltonian"]:
            circ = circuit.entanglement_circuit
        else: 
            circ = circuit.entanglement_circuit_per_qubit
    elif flags.circuit == "strongly_entanglement_circuit":
        weight_shapes = {"weights": (flags.layers, flags.qubits, 3)}
        circ = circuit.strongly_entanglement_circuit

    elif flags.circuit == "Cn_circuit":
        weight_shapes = {"weights": (flags.layers, 3)}

        if flags.task in ["Connectedness", "Bipartiteness", "Connected_plus_Bipartite", "Hamiltonian"]:
            circ = lambda inputs, weights: circuit.Cn_circuit(inputs, weights)
        else:
            circ = lambda inputs, weights: circuit.Cn_circuit_per_qubit(inputs, weights)

    #elif flags.circuit == "Cn_circuit":
    #    weight_shapes = {"weights": (flags.layers, 3)}
    #    circ = lambda inputs, weights: circuit.Cn_circuit(inputs, weights)
    
    elif flags.circuit == "Cn_circuit2":
        weight_shapes = {"weights": (flags.layers, 4)}
        circ = circuit.Cn_circuit2
    elif flags.circuit == "Cn_circuit_with_anti_part":
        weight_shapes = {"weights": (flags.layers, 3)}
        circ = circuit.Cn_circuit_with_anti_part
    elif flags.circuit == "Sn_circuit_with_individual_x_rotations":
        weight_shapes = {"weights": (flags.layers, 3), "r_weights": (flags.rotationlayers, flags.qubits)}
        circ = circuit.Sn_circuit_with_individual_x_rotations
    elif flags.circuit == "Sn_free_parameter_circuit":
        weight_shapes = {"weights_x": (flags.layers, flags.qubits),
                         "weights_y": (flags.layers, flags.qubits),
                         "weights_zz": (flags.layers, math.comb(flags.qubits, 2))}
        circ = circuit.Sn_free_parameter_circuit
    elif flags.circuit == "Energy_circuit":
        weight_shapes = {"weights_z": (flags.layers, flags.qubits),
                         "weights_xx_yy": (flags.layers, math.comb(flags.qubits, 2))}
        circ = circuit.Energy_circuit
    elif flags.circuit == "subgraph_circuit":
        weight_shapes = {"weights_sn": (flags.layers, 3),
                         "weights_strongly_entangled": (flags.layers, flags.qubits, 3)}
        circ = circuit.subgraph_circuit
    else:
        msg = f"Circuit {flags.circuit} is unkown."
        raise ValueError(msg)

    n_parameters = sum(math.prod(shape) for shape in weight_shapes.values())
    print("Number of parameters: ", n_parameters)


    dev = qml.device("default.qubit", wires=flags.qubits)
    qnode = qml.QNode(circ, device=dev, interface="torch")

    base = pathlib.Path(flags.base)
    base_output = pathlib.Path("/home/m/menzell/work/invariant-quantum-circuit/Pennylane/Pennylane_ML/Graph_ML/output/max_cut")

    if flags.task in ["Connectedness", "Bipartiteness", "Connected_plus_Bipartite", "Hamiltonian"]:
        dataset = utils.load_patterns(base / flags.data, flags.qubits)
        predictions, targets, weights = performance.fit(qnode, weight_shapes, dataset, samplings=flags.samplings, epochs=flags.epochs)

        ext = f"GC-{flags.circuit}-{flags.qubits}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"
        pd.DataFrame(targets["train"], columns=["sampling", "target"]).explode("target")\
        .to_csv(base_output / (ext + f"targets-train-{flags.name}.csv"), index=False)
        print('First file saved')
        pd.DataFrame(predictions["train"], columns=["sampling", "epoch", "prediction"]).explode("prediction")\
        .to_csv(base_output / (ext + f"predictions-train-{flags.name}.csv"), index=False)
        print('Second file saved')
        pd.DataFrame(targets["test"], columns=["sampling", "target"]).explode("target")\
        .to_csv(base_output / (ext + f"targets-test-{flags.name}.csv"), index=False)
        print('Third file saved')
        pd.DataFrame(predictions["test"], columns=["sampling", "epoch", "prediction"]).explode("prediction")\
        .to_csv(base_output / (ext + f"predictions-test-{flags.name}.csv"), index=False)
        print('Fourth file saved')

    elif flags.task == "Clique":
        dataset = utils.load_patterns_per_qubit(base / flags.data, flags.qubits)
        predictions, targets, weights = performance_supervised.fit(qnode, weight_shapes, dataset, samplings=flags.samplings, epochs=flags.epochs)

        ext = f"GC-{flags.circuit}-{flags.qubits}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"
        pd.DataFrame(targets["train"], columns=["sampling", "target"]).explode("target")\
        .to_csv(base_output / (ext + f"targets-train-{flags.name}.csv"), index=False)
        print('First file saved')
        pd.DataFrame(predictions["train"], columns=["sampling", "epoch", "prediction"]).explode("prediction")\
        .to_csv(base_output / (ext + f"predictions-train-{flags.name}.csv"), index=False)
        print('Second file saved')
        pd.DataFrame(targets["test"], columns=["sampling", "target"]).explode("target")\
        .to_csv(base_output / (ext + f"targets-test-{flags.name}.csv"), index=False)
        print('Third file saved')
        pd.DataFrame(predictions["test"], columns=["sampling", "epoch", "prediction"]).explode("prediction")\
        .to_csv(base_output / (ext + f"predictions-test-{flags.name}.csv"), index=False)
        print('Fourth file saved')

    elif flags.task == "Max_Cut":
        dataset = utils.load_patterns_per_qubit_max_cut(base / flags.data, flags.qubits)
        predictions, targets, weights = performance_supervised_max_cut.fit(qnode, weight_shapes, dataset, samplings=flags.samplings, epochs=flags.epochs)
        
        ext = f"Max_Cut-{flags.circuit}-{flags.qubits}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"

        df = pd.DataFrame(predictions['train'], columns=["sampling", "epoch", "adjacency", "prediction", "max_cut"])
        df['zipped'] = df.apply(lambda row: list(zip(row['adjacency'], row['prediction'], row['max_cut'])), axis=1)
        df = df.explode('zipped').reset_index(drop=True)
        df[['adjacency', 'prediction', 'max_cut']] = pd.DataFrame(df['zipped'].tolist(), index=df.index) 
        df.drop(columns=['zipped'], inplace=True)
        df.to_csv(base_output / (ext + f"predictions-train-{flags.name}.csv"), index=False) 

        df = pd.DataFrame(predictions['test'], columns=["sampling", "epoch", "adjacency", "prediction", "max_cut"])
        df['zipped'] = df.apply(lambda row: list(zip(row['adjacency'], row['prediction'], row['max_cut'])), axis=1)
        df = df.explode('zipped').reset_index(drop=True)
        df[['adjacency', 'prediction', 'max_cut']] = pd.DataFrame(df['zipped'].tolist(), index=df.index) 
        df.drop(columns=['zipped'], inplace=True)
        df.to_csv(base_output / (ext + f"predictions-test-{flags.name}.csv"), index=False) 

    print("Epochs flag: ", flags.epochs)

    ###### ADD EXTRA SAMPLES FOR CONNECTED ######

    #extra_data = torch.load("/Users/home/qiskit_env/Pennylane/data/graph_connectedness/nodes_8-graphs_10_edge_cases.pt")
    
    #extra_x = extra_data[:, :-1]  # shape: (7, 64)
    #extra_y = extra_data[:, -1]  # shape: (7,)

    #print(extra_data.size()) #(7, 65) --> 7 graphs, for each: 8x8 adjacency + 1 label

    ###### DONE WITH EXTRA SAMPLES FOR CONNECTED ######
