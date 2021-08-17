import networkx as nx

from arct.permutation.general import ApproximateTokenSwapper
from copy import deepcopy
from typing import List, Tuple

Swap = Tuple[int, int]


def demo_approx_token_swapping(in_circuit: nx.Graph,
                               mapping: List[int]) -> List[Swap]:

    permuter = ApproximateTokenSwapper(in_circuit)
    original_mapping = list(in_circuit.nodes())
    print("Original mapping:")
    print(original_mapping)
    print()

    permutation_order = permuter.map(mapping)
    new_mapping = deepcopy(original_mapping)

    for permutation in permutation_order:
        new_mapping[permutation[0]], new_mapping[permutation[1]] = new_mapping[
            permutation[1]], new_mapping[permutation[0]]

    print("New mapping:")
    print(new_mapping)
    print()

    return permutation_order

if __name__ == "__main__":
    demo_circuit = nx.convert_node_labels_to_integers(nx.grid_2d_graph(4, 4))
    mapping = [[node, 15 - node] for node in demo_circuit.nodes()]
    demo_circuit_permutations = demo_approx_token_swapping(
        demo_circuit, mapping)
    print("Permutation order:")
    print(demo_circuit_permutations)
