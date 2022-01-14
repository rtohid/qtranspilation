# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import networkx as nx
import numpy as np

from typing import Tuple


class Grid:

    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.height = num_rows
        self.width = num_cols
        self.shape_ = (self.height, self.width)
        self.graph = self.build_graph()

    def build_graph(self):
        return nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(self.height, self.width))

    def adjacency(self, **kwargs):
        return nx.convert_matrix.to_numpy_array(self.graph, **kwargs)

    def labels(self, **kwargs):
        return np.array(self.graph)

    def size(self) -> int:
        return self.height * self.width

    @property
    def shape(self) -> Tuple:
        return self.shape_

    @shape.setter
    def shape(self) -> Tuple:
        self.shape_ = (self.height, self.width)
