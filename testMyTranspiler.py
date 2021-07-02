from qiskit import *
from matplotlib import pyplot as plt
from utils import build_sample_circuit, define_sample_coupling_map
from simpleTranspiler import *

in_circ = build_sample_circuit()
coupling_map = define_sample_coupling_map()
in_circ.draw(output='mpl')
plt.show()

transpiler = SimpleTranspiler(coupling_map)
out_circ_default = transpiler.transpile(in_circ)
out_circ_default.draw(output='mpl')
plt.show()
