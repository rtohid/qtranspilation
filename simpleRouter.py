from copy import copy
import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler import Layout
from qiskit.circuit.library import SwapGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import *

class SimpleRouter(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a `coupling_map` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    def __init__(self, coupling_map):
        """BasicSwap initializer.

        Args:
            coupling_map (CouplingMap): Directed graph represented a coupling map.
        """
        super().__init__()
        self.coupling_map = coupling_map

    def generate_random_layout(self, *regs):
        perm = np.random.permutation(np.arange(self.coupling_map.size()))
        return Layout.from_intlist(perm[:len(*regs)], *regs)

    def run(self, dag):
        """Run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        new_dag = dag._copy_circuit_metadata()

        if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
            raise TranspilerError('Basic swap runs on physical circuits only')

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError('The layout does not match the amount of qubits in the DAG')

        canonical_register = dag.qregs['q']
        # trivial_layout = self.generate_random_layout(canonical_register)
        # trivial_layout = Layout.generate_trivial_layout(canonical_register)
        trivial_layout = Layout.from_intlist(np.arange(len(self.coupling_map.physical_qubits)), canonical_register)
        current_layout = trivial_layout.copy()

        # print(len(list(dag.serial_layers())))
        for layer in dag.serial_layers():
            subdag = layer['graph']
            next_layout = self.generate_random_layout(canonical_register)
            v2p = current_layout.get_virtual_bits().copy()
            v2p_next = next_layout.get_virtual_bits()
            # print("current_layout = ", v2p)
            # print("next_layout = ", v2p_next)
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)
            for vbit, _ in v2p.items():
                pbit = current_layout.get_virtual_bits()[vbit]
                # physical bit in the next layout
                pbit_next = v2p_next[vbit]
                # print(vbit, ' : ', pbit, ' : ', pbit_next)
                if self.coupling_map.distance(pbit, pbit_next) != 0:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(pbit, pbit_next)
                    for swap in range(len(path) - 1):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(SwapGate(),
                                                        qargs=[qubit_1, qubit_2],
                                                        cargs=[])
                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 1):
                        current_layout.swap(path[swap], path[swap + 1])
            # print("updated current_layout = ", current_layout.get_virtual_bits())
        return new_dag
