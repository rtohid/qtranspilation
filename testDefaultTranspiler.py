from qiskit import *
from matplotlib import pyplot as plt
from utils import build_sample_circuit, define_sample_coupling_map
import sys

opt_level = int(sys.argv[1])

in_circ = build_sample_circuit()
coupling_map = define_sample_coupling_map()
in_circ.draw(output='mpl')
plt.show()

print('optimization level = {}'.format(opt_level))
out_circ_default = transpile(in_circ, coupling_map=coupling_map, optimization_level=opt_level)
out_circ_default.draw(output='mpl')
plt.show()
