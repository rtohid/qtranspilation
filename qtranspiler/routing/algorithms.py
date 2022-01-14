# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import networkx as nx
import numpy as np

from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Tuple

from arct.permutation.general import ApproximateTokenSwapper
from qtranspiler.architectures import Grid

Swap = Tuple[int, int]


def approx_token_swapping(in_circuit: Grid,
                          new_labels: np.ndarray) -> List[Swap]:

    def parallelize_swaps(swaps: List[Tuple]) -> Dict:
        visited = set()
        groups = defaultdict(list)
        group_id = 0
        routing = deepcopy(swaps)
        while routing:
            to_be_routed = deepcopy(routing)
            for pair in to_be_routed:
                if pair[0] not in visited and pair[1] not in visited:
                    visited.add(pair[0])
                    visited.add(pair[1])
                    groups[group_id].append(pair)
                    routing.remove(pair)
            group_id = group_id + 1
            visited.clear()
        return groups
    permuter = ApproximateTokenSwapper(in_circuit.graph)
    mapping = list(zip(in_circuit.labels(), new_labels))
    permutation_order = permuter.map(mapping)
    parallel_permutations = parallelize_swaps(permutation_order)

    return parallel_permutations
