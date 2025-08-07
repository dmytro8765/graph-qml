"""Defines circuits which are used to predict some properties of graphs.

The permutation equivariant layers are constructed by exponentiating the sum of
all permutations of a chosen pauli string for given permutation group.

Note that pennylane uses queuing under the hood
(https://docs.pennylane.ai/en/stable/code/qml_queuing.html) to construct the
circuits, which might seem counter intuitive at first.
"""
from __future__ import annotations

import math
from functools import reduce
from itertools import combinations
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch

NON_BATCHED_INPUT = 1
BATCHED_INPUT = 2


class NotEnoughQubitsError(Exception):
    """Error raised when not enough qubits are being used."""

    def __init__(self, number: int, circuit_name: str) -> None:
        """Initialize a NotEnoughQubitsError class with the number of qubits necessary and the circuit name."""
        super().__init__(f"At least {number} qubits are required for the {circuit_name}.")

class WeightsShapeError(Exception):
    """Error raised when the weights passed as input do not match the shape of the circuit."""

    def __init__(self, weights_shape: str, weights_name: str = "Weights") -> None:
        """Initialize a WeightsShapeError class with the correct weight shape."""
        super().__init__(f"{weights_name} expected to have shape {weights_shape}.")

def Dn_XY_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer consisting of Pauli XYs that commutes with the action of Dn.

    Note that the same weight is used for all matrices.
    """
    if num_qubits < 2:
        raise NotEnoughQubitsError(2, circuit_name="Dn_XY_layer")

    # The sum XYI...I + ... + YI...IX commutes with the sum YXI...I + ... + XI...IY
    if num_qubits == 2:
        # is equal to XY + YX
        qml.exp(sum(qml.X(k) @ qml.Y((k + 1) % num_qubits) for k in range(num_qubits)), coeff=1j * weight)
    else:
        qml.exp(sum(qml.X(k) @ qml.Y((k + 1) % num_qubits) for k in range(num_qubits)), coeff=1j * weight)
        qml.exp(sum(qml.Y(k) @ qml.X((k + 1) % num_qubits) for k in range(num_qubits)), coeff=1j * weight)


def Sn_X_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer consisting of Pauli Xs that commutes with the action of Sn.

    Note that the same weight is used for all matrices.
    """
    for i in range(num_qubits):
        qml.RX(weight, wires=i)


def Sn_Y_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer consisting of Pauli Ys that commutes with the action of Sn.

    Note that the same weight is used for all matrices.
    """
    for i in range(num_qubits):
        qml.RY(weight, wires=i)


def Sn_ZZ_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer consisting of Pauli ZZs that commutes with the action of Sn.

    Note that the same weight is used for all matrices.
    """
    if num_qubits < 2:
    # iterate over [(0, 1), ... (0, num_qubits - 1), ..., (num_qubits - 2, num_qubits - 1)]
        raise NotEnoughQubitsError(2, circuit_name="Sn_ZZ_layer")
    for i, j in combinations(range(num_qubits), 2):
        qml.IsingZZ(weight, wires=[i, j])

def XX_YY_gate(weight: torch.Tensor, wires: list[int]) -> None:
    i, j = wires

    qml.CNOT(wires=[i, j])
    qml.RX(weight, wires=j)
    qml.CNOT(wires=[i, j])

    qml.RZ(np.pi/2, wires = i)
    qml.RZ(np.pi/2, wires = j)

    qml.CNOT(wires=[i, j])
    qml.RX(weight, wires=j)
    qml.CNOT(wires=[i, j])

    qml.RZ(-np.pi/2, wires = i)
    qml.RZ(-np.pi/2, wires = j)

def XX_YY_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    if num_qubits < 2:
        raise ValueError("At least 2 qubits required for XX+YY layer.")

    idx = 0
    for i, j in combinations(range(num_qubits), 2):
        XX_YY_gate(weight[idx], wires=[i, j])
        idx += 1

def Cn_ZZ_layer(*, weight: torch.Tensor, num_qubits: int, spacing: int = 0) -> None:
    """Add a layer consisting of Pauli ZZs that commutes with the action of Cn.

    `spacing` is the number of identity gates between the ZZ gates in the tensor
    product of the starting Pauli string:
        ZI{spacing}ZI* (as a regex expression)
    """
    # We have the same Pauli string for the positions (x, y) and (y, x) of the
    # Pauli ZZ matrices
    if spacing + 1 >= num_qubits:
        msg = "More qubits than the spacing between the ZZ matrices are required."
        raise ValueError(msg)
    pairs = set()
    for i in range(num_qubits):
        x = i
        y = (i + spacing + 1) % num_qubits
        if (y, x) not in pairs:
            pairs.add((x, y))
    for i, j in pairs:
        qml.IsingZZ(weight, wires=[i, j])


def Anti_Cn_ZZ_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer of all 2 body ZZ gates that are not used in the Cn ZZ layer.

    This layer is part of the asymmetric study.
    """
    # iterate over [(0, 1), ... (0, num_qubits - 1), ..., (num_qubits - 2, num_qubits - 1)]
    all_possible_2body_gates_indices = combinations(range(num_qubits), 2)
    cyclic_indices = {(i, (i + 1)) for i in range(num_qubits)}
    non_cyclic_indices = set()
    for i, j in all_possible_2body_gates_indices:
        if (i, j) in cyclic_indices or (j, i) in cyclic_indices:
            # we have the same matrix for the indices (i, j) and (j, i)
            continue
        non_cyclic_indices.add((i, j))
    for i, j in non_cyclic_indices:
        qml.IsingZZ(weight, wires=[i, j])


def RX_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer of RX rotations.

    Note that this layer does commute with the action of a permutation group in
    general.
    """
    if len(weight) != num_qubits:
        weights_shape = f"({num_qubits},)"
        raise WeightsShapeError(weights_shape)
    for i in range(num_qubits):
        qml.RX(weight[i], wires=i)

def RY_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer of RY rotations.

    Note that this layer does commute with the action of a permutation group in
    general.
    """
    if len(weight) != num_qubits:
        weights_shape = f"({num_qubits},)"
        raise WeightsShapeError(weights_shape)
    for i in range(num_qubits):
        qml.RY(weight[i], wires=i)

def RZ_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer of RZ rotations.

    Note that this layer does commute with the action of a permutation group in
    general.
    """
    if len(weight) != num_qubits:
        weights_shape = f"({num_qubits},)"
        raise WeightsShapeError(weights_shape)
    for i in range(num_qubits):
        qml.RZ(weight[i], wires=i)

def ZZ_layer(*, weight: torch.Tensor, num_qubits: int) -> None:
    """Add a layer consisting of Pauli ZZs that commutes with the action of Sn.

    Note that the same weight is used for all matrices.
    """
    if num_qubits < 2:
        raise NotEnoughQubitsError(2, circuit_name="ZZ_layer")
    # iterate over [(0, 1), ... (0, num_qubits - 1), ..., (num_qubits - 2, num_qubits - 1)]
    for indx, (i, j) in enumerate(combinations(range(num_qubits), 2)):
        qml.IsingZZ(weight[indx], wires=[i, j])

################################################################################
# Note: There is no special reason why following methods are seperated.

# Three methods that do almost the same thing could also be summarized into one.
# Due to development, these were created separately and do not cause much
# inconvenience in their application.
def Cn_kbody_X_layer(*, weight: torch.Tensor, k: int, num_qubits: int) -> None:
    """Add a layer consisting of the tensor product of k pauli X matrices that commutes with the action of Cn.

    Note that the same weight is used for all matrices.
    """
    if k == num_qubits:
        qml.exp(gate_prod(qml.PauliX(n) for n in range(num_qubits)), coeff=1j*weight)
    else:
        for indices in (((n + i) % num_qubits for i in range(k)) for n in range(num_qubits)):
          qml.exp(gate_prod(qml.PauliX(index) for index in indices), coeff=1j*weight)


def Cn_kbody_Y_layer(weight: torch.Tensor, k: int, num_qubits: int) -> None:
    """Add a layer consisting of the tensor product of k pauli Y matrices that commutes with the action of Cn.

    Note that the same weight is used for all matrices.
    """
    if k == num_qubits:
        qml.exp(gate_prod(qml.PauliY(n) for n in range(num_qubits)), coeff=1j*weight)
    else:
        for indices in (((n + i) % num_qubits for i in range(k)) for n in range(num_qubits)):
          qml.exp(gate_prod(qml.PauliY(index) for index in indices), coeff=1j*weight)


def Cn_kbody_Z_layer(weight: torch.Tensor, k: int, num_qubits: int) -> None:
    """Add a layer consisting of the tensor product of k pauli Z matrices that commutes with the action of Cn.

    Note that the same weight is used for all matrices.
    """
    if k == num_qubits:
        qml.exp(gate_prod(qml.PauliZ(n) for n in range(num_qubits)), coeff=1j*weight)
    else:
        for indices in (((n + i) % num_qubits for i in range(k)) for n in range(num_qubits)):
          qml.exp(gate_prod(qml.PauliZ(index) for index in indices), coeff=1j*weight)

# End of the three seperated methods.
################################################################################


def graph_state(inputs: torch.Tensor, num_qubits: int) -> None:
    """Embeds the input into a graph state.

    Each qubit represents a node in the graph. First, the qubits are mapped to
    the |+> state. Second, if an edge exists between node i and j, we apply a
    controlled Z Gate.

    It is expected that the n times n adjacency matrix was converted to a
    one dimensional n * n tensor. This is due the fact that a n times n tensor
    is interpreted as a batch of size n with feature vectors of n features by
    pytorch and pennylane.
    """
    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjecency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*inputs[:, i, j], wires=[i, j])


def xx_graph_state(inputs: torch.Tensor, num_qubits: int) -> None:
    """Embeds the input into a graph state used in a commuting XX circuit.

    Each qubit represents a node in the graph. First, the qubits are mapped to
    the |+> state. Second, if an edge exists between node i and j, we apply a
    controlled XX Gate.

    It is expected that the n times n adjacency matrix was converted to a
    one dimensional n * n tensor. This is due the fact that a n times n tensor
    is interpreted as a batch of size n with feature vectors of n features by
    pytorch and pennylane.
    """
    if inputs.ndim == BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == NON_BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i in range(num_qubits):
        qml.Hadamard(wires=i)

    for i, j in combinations(range(num_qubits), 2):
        qml.IsingXX(np.pi*inputs[:, i, j], wires=[i, j])

def Cn_kbody_circuit(inputs:torch.Tensor, weights: torch.Tensor, k: int | list[int] = 2) -> torch.Tensor:
    """Circuit for a problem that is invariant under the action of Cn.

    This circuit allows to use different k-body gates acting on k qubits. If `k`
    is an integer only `k`-body gates are used whereas a list can specify a mix.

    The number of layers is determined by the first dimension of the weights
    tensor.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        k = [2, 3]
        weight_shapes = {"weights": (num_layers, 2 + len(k))}
        Cn_kbody_circuit = lambda inputs, weights: circuit.Cn_kbody_circuit(inputs, weights, k=k)
        qnode = qml.QNode(Cn_kbody_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    kbody_layers = [k] if isinstance(k, int) else k

    if len([k for k in kbody_layers if k < 1]) > 0:
        msg = "Cannot create k body gates with k < 1."
        raise ValueError(msg)

    num_qubits = get_num_qubits_from_inputs(inputs)
    if weights.ndim != 2 or weights.shape[1] != 2 + len(kbody_layers):
        weights_shape = f"(num_layers, {2 + len(kbody_layers)} for {kbody_layers = })"
        raise WeightsShapeError(weights_shape)
    graph_state(inputs, num_qubits=num_qubits)

    num_layers, _ = weights.shape
    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits)
        for i, j in enumerate(kbody_layers):
            Cn_kbody_Z_layer(weight=weights[layer_index, 2 + i], num_qubits=num_qubits, k=j)

    return qml.expval(gate_prod(qml.PauliX(n) for n in range(num_qubits)))


def Cn_kbody_commuting_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Circuit consisting solely of all kbody Pauli X matrices.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We wanted to test what happens if the layers itself are commuting. Applying
    more than one layer has no effect apart from adding the weights. We cannot
    increase the dimension.

    Keep in mind that if the Circuit and the observable commute the circuit has
    no effect at all. We get Tr[UpU'O]=Tr[pU'OU]=Tr[pU'UO]=Tr[pO].

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 1
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits)}
        qnode = qml.QNode(circuit.Cn_kbody_commuting_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    if weights.ndim != 2 or weights.shape[1] != num_qubits:
        weights_shape = "(num_layers, num_qubits)"
        raise WeightsShapeError(weights_shape)
    xx_graph_state(inputs, num_qubits)

    for layer_weights in weights:
        for k, weight in zip(range(1, num_qubits + 1), layer_weights):
            Cn_kbody_X_layer(weight=weight, num_qubits=num_qubits, k=k)

    # Using X@X@...@X the circuit commutes with the observable -> circuit has no effect
    return qml.expval(gate_prod(qml.PauliX(n) for n in range(num_qubits)))


def Cn_kbody_commuting_reference_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Circuit consisting of all kbody Pauli X and Pauli Y matrices.

    A reference circuit where layers itself do not commute.

    The number of layers is determined by the first dimension of the weights
    tensor.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits, 2)}
        qnode = qml.QNode(circuit.Cn_kbody_commuting_reference_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    if weights.ndim != 3 or weights.shape[1] != num_qubits or weights.shape[2] != 2:
        weights_shape = "(num_layers, 2, num_qubits)"
        raise WeightsShapeError(weights_shape)
    xx_graph_state(inputs, num_qubits)

    for layer_weights in weights:
        for k, weight in zip(range(1, num_qubits + 1), layer_weights):
            Cn_kbody_X_layer(weight=weight[0], num_qubits=num_qubits, k=k)
            Cn_kbody_Y_layer(weight=weight[0], num_qubits=num_qubits, k=k)

    return qml.expval(gate_prod(qml.PauliZ(n) for n in range(num_qubits)))


def get_num_qubits_from_inputs(inputs: torch.Tensor) -> int:
    """Return the number of qubits which are needed for embedding the inputs."""
    if inputs.ndim == NON_BATCHED_INPUT:
        dimension, = inputs.shape
        return int(math.sqrt(dimension))
    if inputs.ndim == BATCHED_INPUT:
        _, dimension = inputs.shape
        return int(math.sqrt(dimension))

    msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
    raise ValueError(msg)


def gate_prod(gates: Iterable[qml.operation.Operation]) -> qml.operation.Operation:
    """Return the tensor product of the given gates."""
    return reduce(lambda x, y: x @ y, gates)


def Sn_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create the permutation equivariant circuit for graph predictions.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 3)}
        qnode = qml.QNode(circuit.Sn_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)

    num_layers, _ = weights.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights[layer_index, 2], num_qubits=num_qubits)
    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))

def Sn_circuit_per_qubit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create the permutation equivariant circuit for graph predictions.

    Here, compared to the Sn circuit above: seperate output for each qubit (binary classifiction for each qubit)

    """
    num_qubits = get_num_qubits_from_inputs(inputs)

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)

    num_layers, _ = weights.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights[layer_index, 2], num_qubits=num_qubits)
    return [qml.expval(qml.Z(i)) for i in range(num_qubits)] # per qubit


def Sn_circuit_with_individual_x_rotations(inputs: torch.Tensor, weights: torch.Tensor,
                                           e_weights: torch.Tensor, r_weights: torch.Tensor) -> torch.Tensor:
    """Create the permutation equivariant circuit for graph predictions.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 3), "e_weights": (num_qubits,)}
        qnode = qml.QNode(circuit.Sn_circuit_with_individual_x_rotations, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    if r_weights.ndim != 2 or r_weights.shape[1] != num_qubits:
        weights_shape = "(r_layers, num_qubits)"
        raise WeightsShapeError(weights_shape, weights_name="Rotation weights")
    if e_weights.ndim != 2 or e_weights.shape[0] != r_weights.shape[0] or e_weights.shape[1] != 2:
        weights_shape = "(r_layers, 2)"
        raise WeightsShapeError(weights_shape, weights_name="Extended weights")

    num_layers, _ = weights.shape
    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights[layer_index, 2], num_qubits=num_qubits)

    num_r_layers, _ = r_weights.shape
    for layer_index in range(num_r_layers):
        RX_layer(weight=r_weights[layer_index], num_qubits=num_qubits)
        Sn_Y_layer(weight=e_weights[layer_index, 0], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=e_weights[layer_index, 1], num_qubits=num_qubits)

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))


def entanglement_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create the entanglement circuit for graph predictions.
    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_layers = 5
        num_qubits = 6
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits)}
        qnode = qml.QNode(circuit.entanglement_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```
    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != num_qubits:
        weights_shape = "(num_layers, num_qubits)"
        raise WeightsShapeError(weights_shape)
    qml.BasicEntanglerLayers(weights=weights, wires=range(num_qubits))

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))

def entanglement_circuit_per_qubit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create the entanglement circuit for graph predictions.
    
    Measurement: for each qubit separately, output of dimension # of qubits.
    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != num_qubits:
        weights_shape = "(num_layers, num_qubits)"
        raise WeightsShapeError(weights_shape)
    qml.BasicEntanglerLayers(weights=weights, wires=range(num_qubits))

    return [qml.expval(qml.Z(i)) for i in range(num_qubits)] # per qubit

def Sn_free_parameter_circuit(inputs: torch.Tensor, weights_x: torch.tensor,
                              weights_y: torch.tensor, weights_zz: torch.tensor) -> torch.Tensor:
    """Circuit as the permutation equivariant circuit but with independent parameters.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 3)}
        qnode = qml.QNode(circuit.Sn_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)

    graph_state(inputs, num_qubits)

    weights_shape = "(num_layers, num_qubits)"
    if weights_x.ndim != 2 or weights_x.shape[1] != num_qubits:
        raise WeightsShapeError(weights_shape)
    if weights_y.ndim != 2 or weights_y.shape[1] != num_qubits:
        raise WeightsShapeError(weights_shape)
    if weights_zz.ndim != 2 or weights_zz.shape[1] != math.comb(num_qubits, 2):
        raise WeightsShapeError(weights_shape)
    if(weights_x.shape[0] != weights_y.shape[0] or weights_y.shape[0] != weights_zz.shape[0]):
        msg = "Weights tensors are expected to have the same number of layers"
        raise ValueError(msg)


    num_layers, _ = weights_x.shape

    for layer_index in range(num_layers):
        RX_layer(weight=weights_x[layer_index], num_qubits=num_qubits)
        RY_layer(weight=weights_y[layer_index], num_qubits=num_qubits)
        ZZ_layer(weight=weights_zz[layer_index], num_qubits=num_qubits)
    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))
    #return [qml.expval(qml.Z(i)) for i in range(num_qubits)] # per qubit

def strongly_entanglement_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create the strongly entanglement circuit for graph predictions.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_layers = 5
        num_qubits = 6
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, num_qubits, 3)}
        qnode = qml.QNode(circuit.strongly_entanglement_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    graph_state(inputs, num_qubits)

    weights_shape = "(num_layers, num_qubits, 3)"
    if weights.ndim != 3 or weights.shape[1] != num_qubits or weights.shape[2] != 3:
        raise WeightsShapeError(weights_shape)
    num_layers = weights.shape[0]
    ranges = tuple((layer % (2)) + 1 for layer in range(num_layers))
    qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits), ranges=ranges)

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))


def Cn_circuit(inputs: torch.Tensor, weights: torch.Tensor, spacing: int | list[int] = 0) -> torch.Tensor:
    """Create a circuit which is invariant under cyclic permutations.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        spacing = [0, 1]
        weight_shapes = {"weights": (num_layers, 4)}
        Cn_circuit = lambda inputs, weights: circuit.Cn_circuit(inputs, weights, spacing=spacing)
        qnode = qml.QNode(Cn_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    if isinstance(spacing, int):
        spacing = [spacing]
    if len([space for space in spacing if space < 0]) > 0:
        msg = "Cannot use spacing less than 0."
        raise ValueError(msg)

    num_qubits = get_num_qubits_from_inputs(inputs)
    if num_qubits < 3:
        raise NotEnoughQubitsError(3, "Cn_circuit")

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 2 + len(spacing):
        weights_shape = f"(num_layers, {2 + len(spacing)}) for {spacing = }"
        raise WeightsShapeError(weights_shape)

    num_layers, _ = weights.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits) # also commutes with Cn
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits) # also commutes with Cn
        for i, space in enumerate(spacing):
            Cn_ZZ_layer(weight=weights[layer_index, 2 + i], num_qubits=num_qubits, spacing=space)

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))

def Cn_circuit_per_qubit(inputs: torch.Tensor, weights: torch.Tensor, spacing: int | list[int] = 0) -> torch.Tensor:
    """Create a circuit which is invariant under cyclic permutations.
    
    Measurement: for each qubit separately, output of dimension # of qubits.
    
    """
    if isinstance(spacing, int):
        spacing = [spacing]
    if len([space for space in spacing if space < 0]) > 0:
        msg = "Cannot use spacing less than 0."
        raise ValueError(msg)

    num_qubits = get_num_qubits_from_inputs(inputs)
    if num_qubits < 3:
        raise NotEnoughQubitsError(3, "Cn_circuit")

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 2 + len(spacing):
        weights_shape = f"(num_layers, {2 + len(spacing)}) for {spacing = }"
        raise WeightsShapeError(weights_shape)

    num_layers, _ = weights.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits) # also commutes with Cn
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits) # also commutes with Cn
        for i, space in enumerate(spacing):
            Cn_ZZ_layer(weight=weights[layer_index, 2 + i], num_qubits=num_qubits, spacing=space)

    return [qml.expval(qml.Z(i)) for i in range(num_qubits)] # per qubit


def Cn_circuit2(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create a circuit which is invariant under cyclic permutations using a spacing of 0 and 1.

    This method does only exists for backwards compatibility!

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 4)}
        qnode = qml.QNode(circuit.Cn_circuit2, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    return Cn_circuit(inputs, weights, spacing=[0, 1])


def Cn_circuit_with_anti_part(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Circuit where the ZZ layers are the missing ZZ gates from the standard ZZ Cn layer.

    This circuit is part of studying the effect of unreleated gates on a
    symmetric problem.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 3)}
        qnode = qml.QNode(circuit.Cn_circuit_with_anti_part, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    if num_qubits < 3:
        raise NotEnoughQubitsError(3, "Cn_circuit_with_anti_part")

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)

    # iterate over range(num_layers) instead of weights itself in order to
    # emphasize the number of layers
    num_layers, _ = weights.shape
    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits) # also commutes with Cn
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits) # also commutes with Cn
        Anti_Cn_ZZ_layer(weight=weights[layer_index, 2], num_qubits=num_qubits)

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))


def subgraph_circuit(inputs: torch.Tensor, weights_strongly_entangled: torch.Tensor, weights_Sn: torch.Tensor) -> torch.Tensor:
    """Create a permutation invariant circuit for the subgraph problem.

    It is expected that two distinct graphs are given as inputs.
    The circuit can then be used to predict whether the second (generally smaller) graph is a subgraph of the first graph.

    The circuit consists of a graph encoding layer and a strongly entangling layer,
    followed by a series of permutation invariant layers, which are distinctly applied to both graphs.

    The shape of the weights_strongly_entangled argument is expected to be (num_layers, num_qubits, 3).
    
    The shape of the weights_Sn argument is expected to be (num_layers, 6).
    Per Sn layer, the first three weights are used for the first graph and the last three weights for the second graph.
    """
    num_qubits = get_num_qubits_from_inputs(inputs)

    graph_state(inputs, num_qubits)

    weights_se_shape = "(num_layers, num_qubits, 3)"
    if weights_strongly_entangled.ndim != 3 or weights_strongly_entangled.shape[1] != num_qubits or weights_strongly_entangled.shape[2] != 3:
        raise WeightsShapeError(weights_se_shape)
    num_se_layers = weights_strongly_entangled.shape[0]
    ranges = tuple((layer % (2)) + 1 for layer in range(num_se_layers))
    qml.StronglyEntanglingLayers(weights=weights_strongly_entangled, wires=range(num_qubits), ranges=ranges)

    if weights_Sn.ndim != 2 or weights_Sn.shape[1] != 6:
        weights_Sn_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_Sn_shape)

    num_Sn_layers, _ = weights_Sn.shape

    for layer_index in range(num_Sn_layers):
        Sn_X_layer(weight=weights_Sn[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights_Sn[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights_Sn[layer_index, 2], num_qubits=num_qubits)

    return [qml.expval(gate_prod(qml.Z(i)) for i in range(num_qubits))] 

def subgraph_other_circuit_app1(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 1: 1 + 1+ 2 = 4 trainable parameters.
    #   - X, Y layers: common;
    #   - Sn layer: split;
    #   - inverse graph state layer: absent.

    main_num_qubits = 6
    sub_num_qubits = 4

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4:
        weights_shape = "(num_layers, 4)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app1_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4:
        weights_shape = "(num_layers, 4)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app2(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 2: 1 + 1 + 2 = 4 trainable parameters.
    #   - X, Y layers: common;
    #   - Sn layer: split;
    #   - inverse graph state layer: fixed, outside of the layers.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4:
        weights_shape = "(num_layers, 4)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjecency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app2_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4:
        weights_shape = "(num_layers, 4)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjecency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app3(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    # Approach 3: 
    #   - X, Y layers: common, 2 parameters;
    #   - Sn layer: split, 2 parameters;
    #   - inverse graph state layer: 1 trainable parameter, outside the layers (hence, weights_ent).

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits + sub_num_qubits

    graph_state(inputs, num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4: # (later: add an error for weights_ent shape)
        weights_shape = "(num_layers, 4) + 1"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1
         
    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)


    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # single output based on main graph nodes measurements

def subgraph_other_circuit_app3_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits + sub_num_qubits

    graph_state(inputs, num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 4: # (later: add an error for weights_ent shape)
        weights_shape = "(num_layers, 4) + 1"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1
         
    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)


    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app4(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 4:
    #   - X, Y layers: split, weights[0, 1], weights[2, 3];
    #   - Sn layer: split, weights[4, 5];
    #   - inverse graph state layer: absent.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits + sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app4_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits + sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app5(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 5:
    #   - X, Y layers: split;
    #   - Sn layer: split
    #   - inverse graph state layer: fixed.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjecency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app5_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjecency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app6(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    # Approach 6:
    #   - X, Y layers: split;
    #   - Sn layer: split;
    #   - inverse graph state layer: trainable.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)    

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app6_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)    

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 6:
        weights_shape = "(num_layers, 6)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 4], num_qubits=main_num_qubits)

        for i in range(sub_num_qubits - 1): # apply X, Y gates to the subgraph qubits
                qml.RX(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)
                qml.RY(weights_sn[layer_index, 1], wires=main_num_qubits+i+1)

        for i in range(sub_num_qubits - 1): # apply same Sn block to subgraph qubits
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                qml.RZ(weights_sn[layer_index, 5], wires=main_num_qubits+j+1)
                qml.CNOT(wires=[main_num_qubits+i, main_num_qubits+j+1])
                j += 1

    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app7(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 7:
    #   - X, Y layers: common;
    #   - Sn layer: common;
    #   - inverse graph state layer: absent.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app7_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits + sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app8(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    # Approach 8:
    #   - X, Y layers: common;
    #   - Sn layer: common;
    #   - inverse graph state layer: fixed.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits+sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjacency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app8_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits+sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        # We get I for inputs[a, i, j] = 0 and CZ for inputs[a, i, j] = 1 where
        # inputs[a] is the adjacency matrix for the a'th input graph
        qml.ControlledPhaseShift(np.pi*((1+inputs[:, i, j])%2), wires=[i, j])

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

def subgraph_other_circuit_app9(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    # Approach 9:
    #   - X, Y layers: common;
    #   - Sn layer: common;
    #   - inverse graph state layer: trainable.

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits+sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)


    return qml.expval(gate_prod(qml.Z(index) for index in range(main_num_qubits))) # 1 output based on main nodes measurements

def subgraph_other_circuit_app9_per_qubit(inputs: torch.Tensor, weights_sn: torch.Tensor, weights_ent: torch.Tensor) -> torch.Tensor:

    main_num_qubits = 6
    sub_num_qubits = 4
    num_qubits = main_num_qubits+sub_num_qubits

    graph_state(inputs, main_num_qubits+sub_num_qubits)

    if weights_sn.ndim != 2 or weights_sn.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    
    num_layers, _ = weights_sn.shape

    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=main_num_qubits+sub_num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits+sub_num_qubits)

    '''if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)'''

    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(weights_ent[0], wires=[i, j]) # 5th trainable parameter here (same across all wires)

    return [qml.expval(qml.Z(i)) for i in range(main_num_qubits)] # per qubit

'''def ansatz_layers(weights_sn, main_num_qubits, sub_num_qubits, inputs=None):
    num_qubits = main_num_qubits + sub_num_qubits
    num_layers, _ = weights_sn.shape
    
    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights_sn[layer_index, 0], num_qubits=num_qubits)
        Sn_Y_layer(weight=weights_sn[layer_index, 1], num_qubits=num_qubits)
        Sn_ZZ_layer(weight=weights_sn[layer_index, 2], num_qubits=main_num_qubits)
        for i in range(sub_num_qubits - 1):
            j = i
            while j < sub_num_qubits - 1:
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                qml.RZ(weights_sn[layer_index, 3], wires=main_num_qubits + j + 1)
                qml.CNOT(wires=[main_num_qubits + i, main_num_qubits + j + 1])
                j += 1

    if inputs.ndim == NON_BATCHED_INPUT:
        inputs = inputs.reshape(1, num_qubits, num_qubits)
    elif inputs.ndim == BATCHED_INPUT: # batched input
        inputs = inputs.reshape(inputs.shape[0], num_qubits, num_qubits)
    else:
        msg = "Inputs are expected to be of size (num_batches, num_nodes*num_nodes) or (num_nodes*num_nodes,)."
        raise ValueError(msg)

    for i, j in combinations(range(num_qubits), 2):
        qml.ControlledPhaseShift(np.pi*((1 + inputs[:, i, j]) % 2), wires=[i, j])'''

def Dn_circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Create a circuit which is invariant under dihedral permutations.

    The number of layers is determined by the first dimension of the weights
    tensor.

    We return the expectation value(s) of the supplied observable according to
    the number of batches.

    Example:
        ```
        import pennylane as qml

        from src import circuit, utils

        dataset = utils.load_patterns("./data/graph_connectedness/nodes_6-graphs_3000-edges_5_6_7.pt", num_nodes=6)
        X, Y = dataset.tensors

        num_qubits = 6
        num_layers = 5
        dev = qml.device("default.qubit", wires=num_qubits)
        weight_shapes = {"weights": (num_layers, 3)}
        qnode = qml.QNode(circuit.Dn_circuit, device=dev, interface="torch")

        model = qml.qnn.TorchLayer(qnode, weight_shapes)
        print(model(X[:2]))
        ```

    """
    num_qubits = get_num_qubits_from_inputs(inputs)
    if num_qubits < 3:
        raise NotEnoughQubitsError(3, "Dn_circuit")

    graph_state(inputs, num_qubits)

    if weights.ndim != 2 or weights.shape[1] != 3:
        weights_shape = "(num_layers, 3)"
        raise WeightsShapeError(weights_shape)
    num_layers, _ = weights.shape
    for layer_index in range(num_layers):
        Sn_X_layer(weight=weights[layer_index, 0], num_qubits=num_qubits) # also commutes with Cn
        Sn_Y_layer(weight=weights[layer_index, 1], num_qubits=num_qubits) # also commutes with Cn
        Dn_XY_layer(weight=weights[layer_index, 2], num_qubits=num_qubits) # also commutes with Cn

    return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits)))

def Energy_circuit(inputs: torch.Tensor, weights_z: torch.tensor, weights_xx_yy: torch.tensor) -> torch.Tensor:

    ############ Layer of independent Z rotations and independent xx+yy rotations ############

    num_qubits = get_num_qubits_from_inputs(inputs)

    graph_state(inputs, num_qubits)

    #print(weights_z)
    #print(weights_xx_yy)

    weights_shape = "(num_layers, num_qubits)"
    if weights_z.ndim != 2 or weights_z.shape[1] != num_qubits:
        raise WeightsShapeError(weights_shape)
    if weights_xx_yy.ndim != 2 or weights_xx_yy.shape[1] != math.comb(num_qubits, 2):
        raise WeightsShapeError(weights_shape)
    if(weights_z.shape[0] != weights_xx_yy.shape[0]):
        msg = "Weights tensors are expected to have the same number of layers"
        raise ValueError(msg)

    num_layers, _ = weights_z.shape

    for layer_index in range(num_layers):
        RZ_layer(weight=weights_z[layer_index], num_qubits=num_qubits)
        XX_YY_layer(weight=weights_xx_yy[layer_index], num_qubits=num_qubits)
    #return qml.expval(gate_prod(qml.Z(index) for index in range(num_qubits))) # all qubit-measurements clumped together
    return [qml.expval(qml.Z(i)) for i in range(num_qubits)] # per qubit





