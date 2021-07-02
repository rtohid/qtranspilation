from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import BasicSwap
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.passes import EnlargeWithAncilla
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes import TrivialLayout
from simpleRouter import *

class SimpleTranspiler():
	def __init__(self, coupling_map):
		self.coupling_map = coupling_map

	def transpile(self, in_circ):
		pm = PassManager()
		# 1. add ancilla qbits
		_choose_layout = [TrivialLayout(self.coupling_map)]
		_embed = [FullAncillaAllocation(self.coupling_map), EnlargeWithAncilla(), ApplyLayout()]
		# 2. add routing between two mapping
		_route = [SimpleRouter(self.coupling_map)]
		# 3. add swap
		_swap = [BasicSwap(self.coupling_map)]
		pm.append(_choose_layout)
		pm.append(_embed)
		pm.append(_route)
		pm.append(_swap)
		out_circ = pm.run(in_circ)
		return out_circ
