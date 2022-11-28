# Copyright (c) 2022 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import numpy as np
import networkx as nx

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from matplotlib import pyplot as plt
from os import linesep
from typing import List, TypedDict


class Happiness(Enum):
    HAPPY = 1
    UNHAPPY = 0


@dataclass
class QubitPriority:

    qubit_id: int
    current_node: int
    priority: int


class Circuit(nx.Graph):
    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.routes = list()

    def make_grid(self, num_rows: int, num_cols: int) -> None:
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(num_rows, num_cols))
        self.add_edges_from(graph.edges)
        self.width = num_cols
        self.length = num_rows
        self.is_grid = True

    def assign_qubits(self, qubits: List) -> None:
        self.node_to_qubit_map = {node: qubit for node, qubit in enumerate(qubits)}

    def route(self, debug=True):
        self.routes = PriorityTokenSwap(self).route(debug)
        return self.routes

    def set_node_coordinates(self, coordinates):
        self.node_coordinates = coordinates

    def get_node_coordinates(self, coordinates=None):
        if coordinates:
            self.node_coordinates = coordinates
        elif not hasattr(self, "node_coordinates"):
            try:
                if self.is_grid:
                    self.node_coordinates = [
                        (x // self.width, x % self.length)
                        for x in range(self.number_of_nodes())
                    ]
            except:
                self.node_coordinates = [
                    (x, y) for x, y in nx.circular_layout(self).values()
                ]

    def draw(self, coordinates=None, labels=None) -> None:
        self.get_node_coordinates(coordinates)
        positions = self.node_coordinates
        nx.draw_networkx(self, pos=positions)
        label_positions = {i: (p[0], p[1] + 0.075) for i, p in enumerate(positions)}
        if not labels:
            labels = self.node_to_qubit_map
        nx.draw_networkx_labels(
            self,
            pos=label_positions,
            labels=labels,
            font_color="red",
            font_weight="bold",
        )
        plt.show()


class Swap:
    def __init__(self, qubit_priority: QubitPriority) -> None:
        self.qubit = qubit_priority
        self.happy = []
        self.unhappy = []

    def add_candidate(self, happiness: Happiness, dest):
        if happiness == Happiness.HAPPY:
            self.happy.append(dest)
        if happiness == Happiness.UNHAPPY:
            self.unhappy.append(dest)

    def print_possible_swaps(self):
        print("Happy Qbit Swaps:")
        for swap in self.happy:
            print((self.qubit.qubit_id, swap.qubit_id))
        print("Unhappy Qbit Swaps:")
        for swap in self.unhappy:
            print((self.qubit.qubit_id, swap.qubit_id))


class PriorityTokenSwap:
    def __init__(self, circuit: Circuit) -> None:
        self.circuit = circuit
        self.qubit_priorities = self.find_priorities()
        self.swaps = []

    def find_priorities(self) -> QubitPriority:
        qubit_priorities = {}
        for node, qubit in self.circuit.node_to_qubit_map.items():
            priority = nx.shortest_path_length(self.circuit, qubit, node)
            qubit_priorities[qubit] = QubitPriority(qubit, node, priority)
        return qubit_priorities

    def find_high_priorities(self, filtered_qubits: list):

        max_poprioriy = max(
            self.qubit_priorities.values(),
            key=lambda q: q.priority if q.qubit_id not in filtered_qubits else -1,
        )

        return [q for q in self.qubit_priorities.values() if q == max_poprioriy]

    def find_highest_priority(self):
        highest_priority = max(self.qubit_priorities.values(), key=lambda q: q.priority)
        return highest_priority

    def find_neighbors(self, qubit_priority: QubitPriority) -> List:
        node = qubit_priority.current_node
        canididate_qubits = [
            self.qubit_priorities[self.circuit.node_to_qubit_map[neighbor]]
            for neighbor in self.circuit.neighbors(node)
        ]
        return canididate_qubits

    def quantify_happines(
        self, qubit_priority: QubitPriority, neighbor_priority: QubitPriority
    ) -> int:
        qubit = qubit_priority.qubit_id
        qubit_node = qubit_priority.current_node

        neighbor = neighbor_priority.qubit_id
        neighbor_node = neighbor_priority.current_node

        new_qubit_priority = nx.shortest_path_length(self.circuit, qubit, neighbor_node)
        new_neighbor_priority = nx.shortest_path_length(
            self.circuit, neighbor, qubit_node
        )

        if (new_qubit_priority < qubit_priority.priority) and (
            new_neighbor_priority < neighbor_priority.priority
        ):
            return Happiness.HAPPY
        if (
            new_qubit_priority < qubit_priority.priority
        ) and neighbor_priority.priority == 0:
            return Happiness.UNHAPPY
        return False

    def apply_swap(self, swap: Swap, possible_swaps: List):

        best_swap = None
        for destination in possible_swaps:
            if not best_swap:
                best_swap = destination
                continue
            if destination.priority > best_swap.priority:
                best_swap = destination

        qubit = swap.qubit
        qubit_id = qubit.qubit_id

        best_swap = best_swap
        best_swap_id = best_swap.qubit_id

        # swap the nodes
        qubit_node = best_swap.current_node
        best_swap_node = qubit.current_node
        if self.swaps and self.swaps[-1] == (qubit_node, best_swap_node):
            return False
        else:
            qubit.current_node = qubit_node
            best_swap.current_node = best_swap_node
            self.swaps.append((best_swap_node, qubit_node))

            best_swap.priority = nx.shortest_path_length(
                self.circuit, best_swap_id, best_swap_node
            )
            qubit.priority = nx.shortest_path_length(self.circuit, qubit_id, qubit_node)

            self.circuit.node_to_qubit_map[qubit_node] = qubit_id
            self.circuit.node_to_qubit_map[best_swap_node] = best_swap_id
        return True

    def apply_happy_swap(self):
        filtered_qubits = []
        highs = []
        while len(self.qubit_priorities) != len(filtered_qubits):
            if filtered_qubits:
                filtered_qubits.extend([q.qubit_id for q in highs])
            highs = self.find_high_priorities(filtered_qubits)
            swaps = [Swap(swap_qubit) for swap_qubit in highs]
            for swap in swaps:
                swap_candidates = self.find_neighbors(swap.qubit)
                for swap_candidate in swap_candidates:
                    swap_happiness = self.quantify_happines(swap.qubit, swap_candidate)
                    if swap_happiness:
                        swap.add_candidate(swap_happiness, swap_candidate)
                if swap.happy:
                    if self.apply_swap(swap, swap.happy):
                        return True
            filtered_qubits.extend([q.qubit_id for q in highs])
        return False

    def apply_unhappy_swap(self):
        filtered_qubits = []
        highs = []
        while len(self.qubit_priorities) != len(filtered_qubits):
            if filtered_qubits:
                filtered_qubits.extend([q.qubit_id for q in highs])
            highs = self.find_high_priorities(filtered_qubits)
            swaps = [Swap(swap_qubit) for swap_qubit in highs]
            for swap in swaps:
                swap_candidates = self.find_neighbors(swap.qubit)
                for swap_candidate in swap_candidates:
                    swap_happiness = self.quantify_happines(swap.qubit, swap_candidate)
                    if swap_happiness:
                        swap.add_candidate(swap_happiness, swap_candidate)
                if swap.unhappy:
                    if self.apply_swap(swap, swap.unhappy):
                        return True
            filtered_qubits.extend([q.qubit_id for q in highs])
        return False

    def break_loop(self):
        source = self.find_highest_priority().current_node
        print("***")
        print(
            "node->qubit (i.e., physical->logical) map: ",
            list(self.circuit.node_to_qubit_map.items()),
        )
        print("***")

    def draw(self):
        labels = {
            v.current_node: (q, v.priority) for q, v in self.qubit_priorities.items()
        }

        self.circuit.draw(labels=labels)

    def parallelize_swaps(self) -> TypedDict:
        visited = set()
        groups = defaultdict(list)
        group_id = 0
        routing = deepcopy(self.swaps)
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

    def route(self, debug: bool = True):
        if debug:
            self.draw()
        filtered_qubits = []
        while any(v.priority != 0 for v in self.qubit_priorities.values()):
            print(f"swaps: {self.swaps}")

            if self.apply_happy_swap():
                if debug:
                    self.draw()
                continue
            elif self.apply_unhappy_swap():
                if debug:
                    self.draw()
                continue
            else:
                self.break_loop()
                break

        swaps = self.parallelize_swaps()
        swap_depth = len(swaps)
        swap_size = 0
        for v in swaps.values():
            swap_size = swap_size + len(v)
        return swaps
