"""
Use Qiskit to get pretty visualization of each of the quantum circuits,
used to solve the subgraph problem.
"""

from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import matplotlib.pyplot as plt

# ======== Visualisation parameters ========

custom_style = {
    "displaycolor": {
        "rx_0": ("#ea8282", "#000000"),
        "rx_1": ("#DCCB4F", "#000000"),
        "ry_0": ("#88f17f", "#000000"),
        "ry_1": ("#72e1e5", "#000000"),
        "Sn_main": ("#8787ff", "#000000"),
        "Sn_sub": ("#f6bef5ff", "#000000"),
        "CP": ("#000000", "#FFFFFF"),
        "rx_base_0": ("#88f17f", "#000000"),
        "ry_base_0": ("#88c2ff", "#000000"),
        "rz_base_0": ("#ffaa88", "#000000"),
        "rx_base_1": ("#6fcf69", "#000000"),
        "ry_base_1": ("#66aaff", "#000000"),
        "rz_base_1": ("#ff9666", "#000000"),
        "rx_base_2": ("#57c754", "#000000"),
        "ry_base_2": ("#4491ff", "#000000"),
        "rz_base_2": ("#ff8044", "#000000"),
        "rx_base_3": ("#3fb23f", "#000000"),
        "ry_base_3": ("#2278ff", "#000000"),
        "rz_base_3": ("#ff6b22", "#000000"),
        "rx_base_4": ("#279c2a", "#000000"),
        "ry_base_4": ("#005fff", "#000000"),
        "rz_base_4": ("#ff5500", "#000000"),
        "rx_base_5": ("#116611", "#000000"),
        "ry_base_5": ("#0044aa", "#000000"),
        "rz_base_5": ("#773300", "#000000"),
    }
}

plt.rcParams["figure.dpi"] = 80

# ======== General circuit information ========

main_num_qubits = 4
sub_num_qubits = 2
num_qubits = main_num_qubits + sub_num_qubits

# qr = QuantumRegister(6)
# qc = QuantumCircuit(qr)


def custom_rx(theta, label):
    circ = QuantumCircuit(1, name=label)
    circ.rx(theta, 0)
    return circ.to_gate()


def custom_ry(theta, label):
    circ = QuantumCircuit(1, name=label)
    circ.ry(theta, 0)
    return circ.to_gate()


def custom_rz(theta, label):
    circ = QuantumCircuit(1, name=label)
    circ.ry(theta, 0)
    return circ.to_gate()


def perm_invar_gate_main(theta):
    qc_perm = QuantumCircuit(2, name="Sn_main")
    qc_perm.cx(0, 1)
    qc_perm.rz(theta, 1)
    qc_perm.cx(0, 1)
    return qc_perm.to_gate()


def perm_invar_gate_sub(theta):
    qc_perm = QuantumCircuit(2, name="Sn_sub")
    qc_perm.cx(0, 1)
    qc_perm.rz(theta, 1)
    qc_perm.cx(0, 1)
    return qc_perm.to_gate()


def ent_trainable(theta):
    qc_perm = QuantumCircuit(2, name="CP")
    qc_perm.cp(theta, 0, 1)
    return qc_perm.to_gate()


def subgraph_circ1(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_1")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 3])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    return qc


def subgraph_circ2(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_0")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 3])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 1, 5)
    qc.cp(np.pi, 3, 5)
    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 2, 4)

    return qc


def subgraph_circ3(weights_sn, weights_ent):

    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_0")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 3])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i < j:
                ent_gate = ent_trainable(weights_ent[0])
                qc.append(ent_gate, [qr[i], qr[j]])

    return qc


def subgraph_circ4(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            if qubit < 4:
                rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            else:
                rx_gate = custom_rx(weights_sn[layer_index, 1], "rx_1")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            if qubit < 4:
                ry_gate = custom_ry(weights_sn[layer_index, 2], "ry_0")
            else:
                ry_gate = custom_ry(weights_sn[layer_index, 3], "ry_1")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 4])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 5])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    return qc


def subgraph_circ5(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):

        for qubit in range(num_qubits):
            if qubit < 4:
                rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            else:
                rx_gate = custom_rx(weights_sn[layer_index, 1], "rx_1")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            if qubit < 4:
                ry_gate = custom_ry(weights_sn[layer_index, 2], "ry_0")
            else:
                ry_gate = custom_ry(weights_sn[layer_index, 3], "ry_1")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 3])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 1, 5)
    qc.cp(np.pi, 3, 5)
    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 2, 4)

    return qc


def subgraph_circ6(weights_sn, weights_ent):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            if qubit < 4:
                rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            else:
                rx_gate = custom_rx(weights_sn[layer_index, 1], "rx_1")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            if qubit < 4:
                ry_gate = custom_ry(weights_sn[layer_index, 2], "ry_0")
            else:
                ry_gate = custom_ry(weights_sn[layer_index, 3], "ry_1")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(main_num_qubits):
            for j in range(i + 1, main_num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                perm_gate = perm_invar_gate_sub(weights_sn[layer_index, 3])
                qc.append(
                    perm_gate, [qr[main_num_qubits + i], qr[main_num_qubits + j + 1]]
                )
                j += 1
        qc.barrier()

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i < j:
                ent_gate = ent_trainable(weights_ent[0])
                qc.append(ent_gate, [qr[i], qr[j]])

    return qc


def subgraph_circ7(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_0")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])

    return qc


def subgraph_circ8(weights_sn):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_0")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])
        qc.barrier()

    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 1, 5)
    qc.cp(np.pi, 3, 5)
    qc.cp(np.pi, 0, 1)
    qc.cp(np.pi, 2, 4)

    return qc


def subgraph_circ9(weights_sn, weights_ent):
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rx(weights_sn[layer_index, 0], "rx_0")
            qc.append(rx_gate, [qr[qubit]])
        qc.barrier()

        for qubit in range(num_qubits):
            ry_gate = custom_ry(weights_sn[layer_index, 1], "ry_0")
            qc.append(ry_gate, [qr[qubit]])
        qc.barrier()

        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                perm_gate = perm_invar_gate_main(weights_sn[layer_index, 2])
                qc.append(perm_gate, [qr[i], qr[j]])
        qc.barrier()

    for i in range(num_qubits):
        for j in range(num_qubits):
            if i < j:
                ent_gate = ent_trainable(weights_ent[0])
                qc.append(ent_gate, [qr[i], qr[j]])
    return qc


def subgraph_baseline(weights):
    """
    Approach 1 of strong entanglement:
    - different parameters for RX, RY, RZ layers;
    - different parameters for each qubit inside a RX (RY, RZ) layer;
    - each layer after rotations: entangle each qubit with the neighbour.

    Parameters per layer: 3 * (main + sub), here = 30.
    """

    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    num_layers, _, _ = weights.shape

    for layer_index in range(num_layers):
        for qubit in range(num_qubits):
            rx_gate = custom_rz(weights[layer_index, 0, qubit], f"rx_base_{qubit}")
            qc.append(rx_gate, [qr[qubit]])
            ry_gate = custom_rz(weights[layer_index, 1, qubit], f"ry_base_{qubit}")
            qc.append(ry_gate, [qr[qubit]])
            rz_gate = custom_rz(weights[layer_index, 2, qubit], f"rz_base_{qubit}")
            qc.append(rz_gate, [qr[qubit]])

    """Approach 2 v1: common entangling CNOT layer for ALL qubits."""
    """for i in range(6):
        qc.cx(i, (i + 1) % 6)"""

    """Approach 2 v2: separate entangling CNOT layers for main- and subgraph qubits."""
    for i in range(4):
        qc.cx(i, (i + 1) % 4)

    for i in range(4, 5):
        qc.cx(i, i + 1)

    qc.barrier()

    return qc


# ======== Draw the wanted circuit ========

circ_data = {
    0: {
        "circuit": subgraph_baseline,
        "args": {"weights": np.random.rand(1, 3, 6)},
        "name": "Circuit 0",
    },
    1: {
        "circuit": subgraph_circ1,
        "args": {"weights_sn": np.random.rand(1, 4)},
        "name": "Circuit 1",
    },
    2: {
        "circuit": subgraph_circ2,
        "args": {"weights_sn": np.random.rand(1, 4)},
        "name": "Circuit 2",
    },
    3: {
        "circuit": subgraph_circ3,
        "args": {
            "weights_sn": np.random.rand(1, 4),
            "weights_ent": np.random.rand(
                1,
            ),
        },
        "name": "Circuit 3",
    },
    4: {
        "circuit": subgraph_circ4,
        "args": {"weights_sn": np.random.rand(1, 6)},
        "name": "Circuit 4",
    },
    5: {
        "circuit": subgraph_circ5,
        "args": {"weights_sn": np.random.rand(1, 6)},
        "name": "Circuit 5",
    },
    6: {
        "circuit": subgraph_circ6,
        "args": {
            "weights_sn": np.random.rand(1, 6),
            "weights_ent": np.random.rand(
                1,
            ),
        },
        "name": "Circuit 6",
    },
    7: {
        "circuit": subgraph_circ7,
        "args": {"weights_sn": np.random.rand(1, 3)},
        "name": "Circuit 7",
    },
    8: {
        "circuit": subgraph_circ8,
        "args": {"weights_sn": np.random.rand(1, 3)},
        "name": "Circuit 8",
    },
    9: {
        "circuit": subgraph_circ9,
        "args": {
            "weights_sn": np.random.rand(1, 3),
            "weights_ent": np.random.rand(
                1,
            ),
        },
        "name": "Circuit 9",
    },
}

for i in range(len(circ_data)):

    data = circ_data[i]
    circuit = data["circuit"]
    qc = circuit(**data["args"])

    fig = qc.draw("mpl", style=custom_style)
    fig.suptitle(f"Circuit {i}", fontsize=20)
    fig.savefig(
        f"graph/output/circuit_diagrams/circuit_{i}.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        f"node/output/circuit_diagrams/circuit_{i}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
