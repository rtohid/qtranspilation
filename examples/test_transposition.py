from qtranspiler.architectures import Grid
from qtranspiler.utils import partial_transposition

g = Grid(4,4)
print(partial_transposition(g, 2))


