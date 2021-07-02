from qiskit import *
from qiskit.transpiler import CouplingMap

def build_sample_circuit():
	q = QuantumRegister(4, 'q')
	in_circ = QuantumCircuit(q)
	in_circ.cx(q[0], q[1])
	in_circ.h(q[2])
	in_circ.cx(q[1], q[3])
	in_circ.cx(q[0], q[2])
	in_circ.h(q[1])
	in_circ.cx(q[3], q[2])
	in_circ.cx(q[0], q[1])
	return in_circ

def define_sample_coupling_map():
	coupling = [[0, 1], [1, 2], [0, 3], [1, 4], [2, 5], [3, 4], [4, 5]]
	# coupling = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
	coupling_map = CouplingMap(couplinglist=coupling)
	return coupling_map
