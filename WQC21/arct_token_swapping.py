import networkx as nx
import time

from arct.permutation.general import ApproximateTokenSwapper
from arct.permutation.cartesian import permute_grid
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Tuple

Swap = Tuple[int, int]


def demo_approx_token_swapping(in_circuit: nx.Graph,
                               mapping: List[int]) -> List[Swap]:

    permuter = ApproximateTokenSwapper(in_circuit)
    original_mapping = list(in_circuit.nodes())
    # print("Original mapping:")
    # print(original_mapping)
    # print()

    permutation_order = permuter.map(mapping)
    new_mapping = deepcopy(original_mapping)

    for permutation in permutation_order:
        new_mapping[permutation[0]], new_mapping[permutation[1]] = new_mapping[
            permutation[1]], new_mapping[permutation[0]]

    # print("New mapping:")
    # print(new_mapping)
    # print()

    return permutation_order


def parallelize_swaps(swaps: List[Tuple]) -> Dict:
    visited = set()
    groups = defaultdict(list)
    group_id = 0
    routing = deepcopy(demo_circuit_permutations)
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


if __name__ == "__main__":
    demo_circuit = nx.convert_node_labels_to_integers(nx.grid_2d_graph(12, 3))
    mapping = [[node, 35 - node] for node in demo_circuit.nodes()]
    start = time.process_time()
    demo_circuit_permutations = demo_approx_token_swapping(
        demo_circuit, mapping)
    time_local = time.process_time() - start
    print(f"Finding permutation: {time_local}")
    # print("Permutation order:")
    # print(demo_circuit_permutations)
    start = time.process_time()
    groups = parallelize_swaps(demo_circuit_permutations)
    time_token_swap = time.process_time() - start
    print(f"Parallelizing permutation: {time_token_swap}")
    print(f"speedup: {time_token_swap/time_local}")
    # print(groups)
