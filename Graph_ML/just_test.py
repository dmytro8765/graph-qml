"""Main file used to run experiments and save the results."""

import argparse
import math
import pathlib
import sys

import pandas as pd
import pennylane as qml
import torch
import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import FakeTorino
import logging

from src import circuit, performance, utils, performance_supervised, performance_supervised_max_cut
from credentials import TOKEN, CRN

torch.manual_seed(123)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layers", help="Number of layers", default=30, type=int)
    parser.add_argument("-rl", "--rotationlayers", help="Number of layers with individual rotations", default=0,
                        type=int)
    parser.add_argument("-q", "--qubits", help="Number of qubits", default=8, type=int)
    parser.add_argument("-n", "--name", help="Number to add to filenames to prevent overrides", default=3, type=int)
    parser.add_argument("-b", "--base", help="Base directory", default="/Users/danielle/BAIQO/Pennylane_ML", type=str)
    parser.add_argument("-c", "--circuit", help="Select circuit to run", default="Cn_circuit")
    parser.add_argument("-t", "--task", help="Select task to be trained for", default="Connectedness")
    parser.add_argument("-d", "--data", help="filename for dataset", default="nodes_8-graphs_3000.pt",
                        type=str)
    parser.add_argument("-e", "--epochs", help="number of epochs", default=50, type=int)
    parser.add_argument("-s", "--samplings", help="number of samplings", default=10, type=int)
    parser.add_argument("-w", "--weights", help="filename weights of trained model to use", type=str)
    parser.add_argument("-sn", "--sample_number", help="number of sample used for trained model", type=int)


    flags = parser.parse_args()

    weight_shapes = {"weights": (flags.layers, 3)}
    circ = lambda inputs, weights: circuit.Cn_circuit(inputs, weights)

    n_parameters = sum(math.prod(shape) for shape in weight_shapes.values())
    print("Number of parameters: ", n_parameters)

    # Enable Qiskit Runtime logging
    logging.getLogger('qiskit_ibm_runtime').setLevel(logging.DEBUG)

    service = QiskitRuntimeService.save_account(token=TOKEN, instance=CRN, set_as_default=True, overwrite=True)
    service = QiskitRuntimeService()
    print(service.backends())
    #ibm_backend = service.backend("ibm_kingston")
    ibm_backend = FakeTorino()
    # dev = qml.device("default.qubit", wires=flags.qubits)
    dev = qml.device("qiskit.remote", wires=flags.qubits, backend=ibm_backend, seed_transpiler=42, seed_estimator=42, shots=100, optimization_level=1, dynamical_decoupling={'enable': True}, resilience_level=0, log_level='DEBUG')
    rng = np.random.default_rng(seed=42)
    qnode = qml.QNode(circ, device=dev, interface="torch", diff_method="spsa", gradient_kwargs={'sampler_rng': rng})
    # qnode = qml.QNode(circ, device=dev, interface="torch", gradient_kwargs={'sampler_rng': rng})

    base = pathlib.Path(flags.base)
    dataset = utils.load_patterns(f"{base}/Graph_ML/data/graph_connectedness/{flags.data}",
                                  flags.qubits)
    folder = "connectedness"
    base_output = pathlib.Path(f"{flags.base}/Graph_ML/output/{folder}")
    ext = f"GC-{flags.circuit}-{flags.qubits}-{n_parameters}-sampling_{flags.samplings}-epochs_{flags.epochs}-"
    file_names = {"predictions-test": base_output / (ext + f"predictions-test-{flags.name}.csv"),
                  "targets-test": base_output / (ext + f"targets-test-{flags.name}.csv"),
                  "weights": base_output / f"{flags.weights}"}
    pd.DataFrame(columns=["sampling", "epoch", "prediction"]).to_csv(
        file_names["predictions-test"], index=False
    )
    pd.DataFrame(columns=["sampling", "target"]).to_csv(
        file_names["targets-test"], index=False
    )
    performance.test(qnode, weight_shapes, dataset, samplings=flags.samplings,
                                                    epoch=flags.epochs, file_names=file_names,
                                                    sample_number=flags.sample_number, device=dev)
    print("Run finished!")