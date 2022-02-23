# Copyright (c) 2021 Xin Liang
# Copyright (c) 2021 R. Tohid (@rtohid)
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import networkx as nx
import numpy as np

from qtranspiler.architectures import Grid


def check_inverted(dst_x, dst_y):
    return (dst_x > dst_y)


def line_routing(src, dst):
    """Perform line routing.
    
    @param src: source indices
    @param dst: destination indices
    """

    if len(src.shape) != 1:
        raise ValueError("Source shape must be 1.")
    if len(src) != len(dst):
        raise ValueError("Source and destination must be of the same length.")

    src_length = len(src)
    mapping = np.argsort(dst)
    start = 0
    current = src.copy()
    iteration = 0
    swap_edges = []
    while np.count_nonzero(current == dst) != src_length:
        # check edges
        swap_edge = []
        for i in range(start, src_length - 1, 2):
            if check_inverted(mapping[current[i]], mapping[current[i + 1]]):
                # swap
                tmp = current[i]
                current[i] = current[i + 1]
                current[i + 1] = tmp
                swap_edge.append([i, i + 1])
        if len(swap_edge) > 0:
            swap_edges.append(swap_edge)
        iteration += 1
        start = 1 - start
    return swap_edges


def parallelize_swap_gates(swap_edges, m, n, col=True):
    """Given list of swap_edges for each row/column, reorder them for parallel execution (transposition).

    @params swap_edges: swap_edges collected by line_routing
    @params m, n: m x n grid
    @params col: column routing or row routing. Used to identify position of physical qbit
    
    return swap gates on physical qubit in order
    """

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
                    dst_row[required_row_ind, required_src]
                ])
        matchings.append(matching)


def round_1_column_routing_with_localism(dst_row, dst_column):
    """determine the intermediate_mapping: routing destination in the first round."""
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
    sorted_distance = np.sort(distance.reshape([-1]))
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
                if distance[j, k] <= delta:
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
    return intermediate_mapping, p_swap_edges


def round_1_column_routing(dst_column):
    """Determine the intermediate_mapping: routing destination in the first round."""

    num_rows, num_cols = np.shape(dst_column)
    intermediate_mapping = np.zeros([num_rows, num_cols], dtype=np.int32) - 1

    # initialize how many destinations are available
    # available_dst[i, j]: available dst in column i for dst j
    available_dst = np.zeros([num_cols, num_cols], dtype=np.int32)

    for i in range(num_rows):
        for j in range(num_cols):
            available_dst[j, dst_column[i, j]] += 1

    for i in range(num_rows):
        # build flow network
        # source 0
        # s: 1 ~ n
        # t: n+1 ~ 2n
        # sink: 2n+1
        G = nx.DiGraph()

        # add source and sink edges
        for j in range(num_cols):
            G.add_edge(0, j + 1, capacity=1, weight=1)
            G.add_edge(num_cols + 1 + j,
                       2 * num_cols + 1,
                       capacity=1,
                       weight=1)

        # add bipartitie connection
        for j in range(num_rows):
            for k in range(num_cols):
                if available_dst[k, dst_column[j, k]] > 0:
                    G.add_edge(1 + k,
                               num_cols + 1 + dst_column[j, k],
                               capacity=1,
                               weight=1)

        mincostFlow = nx.max_flow_min_cost(G, 0, 2 * num_cols + 1)

        # find i-th matching: map the selected elements from current row to the i-th row
        for u in mincostFlow:
            if u == 0:
                continue
            vertices = mincostFlow[u]
            for v in vertices:
                if v == 2 * num_cols + 1 or vertices[v] == 0:
                    continue
                # edge (u, v) is in the matching, i.e., qbit will move from column u to v
                # find required source column
                required_src = u - 1
                # find required destination column
                required_dst = v - (num_cols + 1)
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
    for i in range(num_cols):
        swap_edges.append(
            line_routing(np.arange(num_rows), intermediate_mapping[:, i]))
    # adjust order
    p_swap_edges = parallelize_swap_gates(swap_edges, num_rows, num_cols)
    return intermediate_mapping, p_swap_edges


def round_2_row_routing(dst_column):
    """route dst_column to the correct place."""
    num_rows, num_cols = np.shape(dst_column)
    intermediate_mapping = np.zeros([num_rows, num_cols], dtype=np.int32)
    swap_edges = []
    for i in range(num_rows):
        swap_edges.append(line_routing(dst_column[i, :], np.arange(num_cols)))
        intermediate_mapping[i, dst_column[i, :]] = np.arange(num_cols)
    p_swap_edges = parallelize_swap_gates(swap_edges, num_rows, num_cols,
                                          False)
    return intermediate_mapping, p_swap_edges


def round_3_column_routing(dst_row):
    """rout dst_row to the correct place."""
    m, n = np.shape(dst_row)
    intermediate_mapping = np.zeros([m, n], dtype=np.int32)
    swap_edges = []
    for i in range(n):
        swap_edges.append(line_routing(dst_row[:, i], np.arange(m)))
        intermediate_mapping[dst_row[:, i], i] = np.arange(m)
    p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
    return intermediate_mapping, p_swap_edges


def grid_route(src: np.ndarray, dst: np.ndarray, local=True):

    if len(src.shape) != 2:
        raise ValueError(
            f"Invalid grid dimensions ({src.shape}). Expecting 2d grids.")

    if src.size != dst.size:
        err = "Source and destination must be of the same dimension.\n   "
        err = err + f"Source shape ({src.shape}), destination shape({dst.shape})"
        raise ValueError(err)

    swap_gates = list()

    num_rows, num_cols = src.shape

    arg_src = np.argsort(src.reshape([-1]))
    arg_dst = np.argsort(dst.reshape([-1]))
    mapping = np.zeros(src.size, dtype=np.int32)
    mapping[arg_src] = arg_dst

    dst_column = mapping.reshape([num_rows, num_cols]) % num_cols
    dst_row = mapping.reshape([num_rows, num_cols]) // num_cols

    if local:
        intermediate_mapping, swaps_1 = round_1_column_routing_with_localism(
            dst_row.copy(), dst_column.copy())
    else:
        intermediate_mapping, swaps_1 = round_1_column_routing(
            dst_column.copy())
    # swap dst_column and dst_row based on the intermediate_mapping
    for i in range(num_cols):
        tmp = dst_column[:, i]
        dst_column[:, i] = tmp[intermediate_mapping[:, i]]
        tmp = dst_row[:, i]
        dst_row[:, i] = tmp[intermediate_mapping[:, i]]
    intermediate_mapping, swaps_2 = round_2_row_routing(dst_column.copy())
    # swap dst_column and dst_row
    for i in range(num_rows):
        tmp = dst_column[i, :]
        dst_column[i, :] = tmp[intermediate_mapping[i, :]]
        tmp = dst_row[i, :]
        dst_row[i, :] = tmp[intermediate_mapping[i, :]]
    intermediate_mapping, swaps_3 = round_3_column_routing(dst_row.copy())
    # swap dst_column and dst_row based on the intermediate_mapping
    for i in range(num_cols):
        tmp = dst_column[:, i]
        dst_column[:, i] = tmp[intermediate_mapping[:, i]]
        tmp = dst_row[:, i]
        dst_row[:, i] = tmp[intermediate_mapping[:, i]]
    swap_gates = swaps_1 + swaps_2 + swaps_3
    return swap_gates


def grid_route_two_directions(src, dst, local=True):
    m, n = np.shape(src)
    swap_edges_1 = grid_route(src, dst, local)
    swap_edges_2 = grid_route(np.transpose(src), np.transpose(dst), local)
    if len(swap_edges_1) <= len(swap_edges_2):
        return swap_edges_1
    else:
        # transpose swap gates
        for lis in swap_edges_2:
            for pair in lis:
                pair[0] = (pair[0] % m) * n + (pair[0] // m)
                pair[1] = (pair[1] % m) * n + (pair[1] // m)
        return swap_edges_2
