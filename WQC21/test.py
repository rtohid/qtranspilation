import networkx as nx
import numpy as np
import time

from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt
from typing import Dict, List, Tuple

from arct.permutation.general import ApproximateTokenSwapper

grid_dimensions = [(2, 2), (3, 2), (3, 3), (4, 4), (8, 2), (8, 4), (16, 2),
                   (8, 8), (16, 4), (32, 2), (8, 16), (32, 4), (64, 2),
                   (16, 16), (32, 8), (64, 4), (128, 2)]

Swap = Tuple[int, int]


def check_inverted(dst_x, dst_y):
    return (dst_x > dst_y)


# perform line routing
# @param src: source indices
# @param dst: destination indices
def line_routing(src, dst):
    assert (len(src.shape) == 1)
    assert (len(src) == len(dst))
    n = len(src)
    mapping = np.argsort(dst)
    start = 0
    current = src.copy()
    iteration = 0
    swap_edges = []
    while np.count_nonzero(current == dst) != n:
        # check edges
        swap_edge = []
        for i in range(start, n - 1, 2):
            if check_inverted(mapping[current[i]], mapping[current[i + 1]]):
                # swap
                tmp = current[i]
                current[i] = current[i + 1]
                current[i + 1] = tmp
                swap_edge.append([i, i + 1])
        swap_edges.append(swap_edge)
        iteration += 1
        start = 1 - start
    return swap_edges


# Given list of swap_edges for each row/column, reorder them for parallel execution (transposition)
# @params swap_edges: swap_edges collected by line_routing
# @params m, n: m x n grid
# @params col: column routing or row routing. Used to identify position of physical qbit
# return swap gates on physical qubit in order
def parallelize_swap_gates(swap_edges, m, n, col=True):
    swap_edges_in_parallel = []
    if col:
        num = n
    else:
        num = m
    size = np.zeros([num], dtype=np.int32)
    index = np.zeros([num], dtype=np.int32)
    for i in range(num):
        size[i] = len(swap_edges[i])
    while np.max(size - index) != 0:
        swap_edges_iter = []
        for i in range(num):
            if index[i] < size[i]:
                for edge in swap_edges[i][index[i]]:
                    new_edge = edge.copy()
                    if col:
                        for j in range(len(new_edge)):
                            new_edge[j] = edge[j] * n + i
                    else:
                        for j in range(len(new_edge)):
                            new_edge[j] = i * n + edge[j]
                    swap_edges_iter.append(new_edge)
                index[i] += 1
        swap_edges_in_parallel.append(swap_edges_iter)
    return swap_edges_in_parallel


def round_1_column_routing(dst_column):
    # determine the intermediate_mapping: routing desitation in the first round
    m, n = np.shape(dst_column)
    intermediate_mapping = np.zeros([m, n], dtype=np.int32) - 1
    # initiatilize how many destinations are available
    # available_dst[i, j]: available dst in column i for dst j
    available_dst = np.zeros([n, n], dtype=np.int32)
    for i in range(m):
        for j in range(n):
            available_dst[j, dst_column[i, j]] += 1
    for i in range(m):
        # build flow network
        # source 0
        # s: 1 ~ n
        # t: n+1 ~ 2n
        # sink: 2n+1
        G = nx.DiGraph()
        # add source and sink edges
        for j in range(n):
            G.add_edge(0, j + 1, capacity=1, weight=1)
            G.add_edge(n + 1 + j, 2 * n + 1, capacity=1, weight=1)
        # add bipartitie connection
        for j in range(m):
            for k in range(n):
                if available_dst[k, dst_column[j, k]] > 0:
                    G.add_edge(1 + k,
                               n + 1 + dst_column[j, k],
                               capacity=1,
                               weight=1)
        # nx.draw(G, with_labels = True)
        # plt.show()
        mincostFlow = nx.max_flow_min_cost(G, 0, 2 * n + 1)
        # find i-th matching: map the selected elements from current row to the i-th row
        for u in mincostFlow:
            if u == 0:
                continue
            vertices = mincostFlow[u]
            for v in vertices:
                if v == 2 * n + 1 or vertices[v] == 0:
                    continue
                # edge (u, v) is in the matching, i.e., qbit will move from column u to v
                # find required source column
                required_src = u - 1
                # find required destination column
                required_dst = v - (n + 1)
                # find row index in the required source column
                required_row_ind = np.where(
                    dst_column[:, required_src] == required_dst)[0][0]
                # decrement available destination from src to dst
                available_dst[required_src, required_dst] -= 1
                # mark the corresponding data as mapped
                dst_column[required_row_ind, required_src] = -1
                # record the mapping
                intermediate_mapping[i, required_src] = required_row_ind
    # perform column routing
    swap_edges = []
    for i in range(n):
        swap_edges.append(
            line_routing(np.arange(m), intermediate_mapping[:, i]))
    # adjust order
    p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
    return intermediate_mapping


def find_perfect_matching(dst_row, dst_column, current_row, matchings):
    m, n = np.shape(dst_column)
    available_dst = np.zeros([n, n], dtype=np.int32)
    for i in range(m):
        for j in range(n):
            # check if that point is used
            if dst_column[i, j] >= 0:
                available_dst[j, dst_column[i, j]] += 1
    for i in range(m):
        # build flow network
        # source 0
        # s: 1 ~ n
        # t: n+1 ~ 2n
        # sink: 2n+1
        G = nx.DiGraph()
        # add source and sink edges
        for j in range(n):
            G.add_edge(0, j + 1, capacity=1, weight=1)
            G.add_edge(n + 1 + j, 2 * n + 1, capacity=1, weight=1)
        # add bipartitie connection
        for j in range(m):
            for k in range(n):
                if dst_column[j, k] != -1 and available_dst[k,
                                                            dst_column[j,
                                                                       k]] > 0:
                    G.add_edge(1 + k,
                               n + 1 + dst_column[j, k],
                               capacity=1,
                               weight=1)
        # nx.draw(G, with_labels = True)
        # plt.show()
        flowValue, maxFlow = nx.algorithms.flow.maximum_flow(G, 0, 2 * n + 1)
        if flowValue < n:
            # no perfect matching
            break
        # add the perfect matching
        matching = []
        for u in maxFlow:
            if u == 0:
                continue
            vertices = maxFlow[u]
            for v in vertices:
                if v == 2 * n + 1 or vertices[v] == 0:
                    continue
                # edge (u, v) is in the matching, i.e., qbit will move from column u to v
                # find required source column
                required_src = u - 1
                # find required destination column
                required_dst = v - (n + 1)
                # find row index in the required source column
                required_row_ind = np.where(
                    dst_column[:, required_src] == required_dst)[0][0]
                # decrement available destination from src to dst
                available_dst[required_src, required_dst] -= 1
                # mark the corresponding data as mapped
                dst_column[required_row_ind, required_src] = -1
                # record matching [j, j', i, i']
                matching.append([
                    required_src, required_dst, current_row + required_row_ind,
                    current_row + dst_row[required_row_ind, required_src]
                ])
        matchings.append(matching)


def round_1_column_routing_with_localism(dst_row, dst_column):
    # determine the intermediate_mapping: routing desitation in the first round
    m, n = np.shape(dst_column)
    intermediate_mapping = np.zeros([m, n], dtype=np.int32) - 1
    matchings = []
    window_size = 1
    while len(matchings) < m and window_size <= m * 2:
        # iterate over each slice
        start = 0
        for i in range(m // window_size + 1):
            end = np.min([start + window_size, m])
            if start < end:
                find_perfect_matching(dst_row[start:end],
                                      dst_column[start:end], start, matchings)
                start = end
        window_size = 2 * window_size
    assert (len(matchings) == m)
    # bottleneck bipartite perfect matching
    distance = np.zeros([m, m])
    for j in range(m):
        matching = matchings[j]
        for k in range(m):
            # compute weight for j-th matching to k-th row
            dist = 0
            for i in range(n):
                dist += np.abs(matching[i][2] - k) + np.abs(matching[i][3] - k)
            distance[j, k] = dist
    # binary search based BBPM
    bottleneck_matching = []
    sorted_distance = np.sort(distance).reshape([-1])
    while sorted_distance.size != 0:
        mid = sorted_distance.size // 2
        delta = sorted_distance[mid]
        # create complete bipartite graph H
        H = nx.DiGraph()
        # build flow network
        # source 0
        # s: 1 ~ m
        # t: m+1 ~ 2m
        # sink: 2m+1
        # add source and sink edges
        for j in range(m):
            H.add_edge(0, j + 1, capacity=1, weight=1)
            H.add_edge(m + 1 + j, 2 * m + 1, capacity=1, weight=1)
        # add bipartitie connection
        for j in range(m):
            for k in range(m):
                if distance[j, k] < delta:
                    H.add_edge(1 + j,
                               m + 1 + k,
                               capacity=1,
                               weight=distance[j, k])
        # find matching
        flowValue, maxFlow = nx.algorithms.flow.maximum_flow(H, 0, 2 * m + 1)
        if flowValue < m:
            # no perfect matching
            sorted_distance = sorted_distance[mid + 1:]
        else:
            # perfect matching found
            # record matching
            bottleneck_matching.clear()
            for u in maxFlow:
                if u == 0:
                    continue
                vertices = maxFlow[u]
                for v in vertices:
                    if v == 2 * m + 1 or vertices[v] == 0:
                        continue
                    # find index of matching
                    matching_ind = u - 1
                    # find index of row
                    row_ind = v - (m + 1)
                    bottleneck_matching.append([matching_ind, row_ind])
            sorted_distance = sorted_distance[:mid]
    # test correct result
    # assign mapping
    # intermediate_mapping[i, required_src] = required_row_ind
    for i in range(len(bottleneck_matching)):
        matching = matchings[bottleneck_matching[i][0]]
        row = bottleneck_matching[i][1]
        # map matching to row
        for j in range(len(matching)):
            intermediate_mapping[row, matching[j][0]] = matching[j][2]
    # perform column routing
    swap_edges = []
    for i in range(n):
        swap_edges.append(
            line_routing(np.arange(m), intermediate_mapping[:, i]))
    # adjust order
    p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
    return intermediate_mapping


# route dst_column to the correct place
def round_2_row_routing(dst_column):
    m, n = np.shape(dst_column)
    intermediate_mapping = np.zeros([m, n], dtype=np.int32)
    swap_edges = []
    for i in range(m):
        swap_edges.append(line_routing(dst_column[i, :], np.arange(n)))
        intermediate_mapping[i, dst_column[i, :]] = np.arange(n)
    p_swap_edges = parallelize_swap_gates(swap_edges, m, n, False)
    return intermediate_mapping


# rout dst_row to the correct place
def round_3_column_routing(dst_row):
    m, n = np.shape(dst_row)
    intermediate_mapping = np.zeros([m, n], dtype=np.int32)
    swap_edges = []
    for i in range(n):
        swap_edges.append(line_routing(dst_row[:, i], np.arange(m)))
        intermediate_mapping[dst_row[:, i], i] = np.arange(m)
    p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
    return intermediate_mapping


def grid_route(src, dst, local=False):
    assert (len(src.shape) == 2)
    assert (len(src) == len(dst))
    # create mapping
    m, n = np.shape(src)
    arg_src = np.argsort(src.reshape([-1]))
    arg_dst = np.argsort(dst.reshape([-1]))
    mapping = np.zeros([m * n], dtype=np.int32)
    mapping[arg_src] = arg_dst
    dst_column = mapping.reshape([m, n]) % n
    dst_row = mapping.reshape([m, n]) // n
    if local:
        intermediate_mapping = round_1_column_routing_with_localism(
            dst_row.copy(), dst_column.copy())
    else:
        intermediate_mapping = round_1_column_routing(dst_column.copy())
    # swap dst_column and dst_row based on the intermediate_mapping
    for i in range(n):
        tmp = dst_column[:, i]
        dst_column[:, i] = tmp[intermediate_mapping[:, i]]
        tmp = dst_row[:, i]
        dst_row[:, i] = tmp[intermediate_mapping[:, i]]
    intermediate_mapping = round_2_row_routing(dst_column.copy())
    # swap dst_column and dst_row
    for i in range(m):
        tmp = dst_column[i, :]
        dst_column[i, :] = tmp[intermediate_mapping[i, :]]
        tmp = dst_row[i, :]
        dst_row[i, :] = tmp[intermediate_mapping[i, :]]
    intermediate_mapping = round_3_column_routing(dst_row.copy())
    # swap dst_column and dst_row based on the intermediate_mapping
    for i in range(n):
        tmp = dst_column[:, i]
        dst_column[:, i] = tmp[intermediate_mapping[:, i]]
        tmp = dst_row[:, i]
        dst_row[:, i] = tmp[intermediate_mapping[:, i]]


def random_map(shape: Tuple) -> np.ndarray:
    size = shape[0] * shape[1]
    source = np.arange(size).reshape(shape)
    target = np.arange(size)
    np.random.shuffle(target)
    return (source, target.reshape(shape))


def local_randomizer(map: np.ndarray, depth: int) -> np.ndarray:
    for i in range(0, len(map), depth):
        h, w = np.shape(map)
        map[i:i + depth] = np.random.permutation(
            map.flatten()[h * i:h * (i + depth)]).reshape(depth, w)


class Grid:
    def __init__(self, num_rows: int, num_cols: int) -> None:
        self.height = num_rows
        self.width = num_cols
        self.map = None
        self.graph = None

    def build_graph(self):
        return nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(self.height, self.width))

    def size(self) -> int:
        return self.height * self.width

    def shape(self) -> Tuple:
        return (self.height, self.width)

    def route(self, mapping: np.ndarray, local: int = True):
        return grid_route(self.map, mapping, local)


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


grids = [random_map(grid) for grid in grid_dimensions]


def approx_token_swapping(in_circuit: nx.Graph,
                          mapping: List[int]) -> List[Swap]:

    permuter = ApproximateTokenSwapper(in_circuit)
    permutation_order = permuter.map(mapping)
    parallel_permutations = parallelize_swaps(permutation_order)

    return parallel_permutations


speed_ups = list()
# grids = np.load('grids.npy', allow_pickle='TRUE')
for idx, maps in enumerate(grids):
    source = maps[0]
    target = maps[1]
    demo_circuit = nx.convert_node_labels_to_integers(
        nx.grid_2d_graph(*grid_dimensions[idx]))

    print(grid_dimensions[idx])

    start = time.process_time()
    grid_route(source, target, True)
    time_local = time.process_time() - start
    print(f"local: {time_local}")

    start = time.process_time()
    demo_circuit_permutations = approx_token_swapping(
        demo_circuit, list((zip(source.flatten(), target.flatten()))))
    time_token_swap = time.process_time() - start
    print(f"token swap: {time_token_swap}")
    #
    print(f"speedup: {time_token_swap/time_local}")
    speed_ups.append(time_token_swap / time_local)
print(speed_ups)

# np.save('grids.npy', grids)
