import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def check_inverted(dst_x, dst_y):
	return (dst_x > dst_y)

# perform line routing
# @param src: source indices
# @param dst: destination indices
def line_routing(src, dst, verbose=False):
	assert(len(src.shape) == 1)
	assert(len(src) == len(dst))
	n = len(src)
	mapping = np.argsort(dst)
	if verbose:
		print("start position  = {}".format(src))
		print("target position = {}".format(dst))
	start = 0
	current = src.copy()
	iteration = 0
	swap_edges = []
	while np.count_nonzero(current == dst) != n:
		# check edges
		swap_edge = []
		# odd or even first?
		for i in range(start, n - 1, 2):
			if check_inverted(mapping[current[i]], mapping[current[i+1]]):
				# swap
				tmp = current[i]
				current[i] = current[i+1]
				current[i+1] = tmp
				swap_edge.append([i, i+1])
		if len(swap_edge) > 0:
			swap_edges.append(swap_edge)
		# print("swap_edges in iteration {}: {}".format(iteration, swap_edge))
		# print("current position = {}".format(current))
		iteration += 1
		start = 1 - start
	if verbose:
		print(swap_edges)		
	return swap_edges

# Given list of swap_edges for each row/column, reorder them for parallel execution (transposition)
# @params swap_edges: swap_edges collected by line_routing
# @params m, n: m x n grid
# @params col: column routing or row routing. Used to identify position of physical qbit
# return swap gates on physical qubit in order 
def parallelize_swap_gates(swap_edges, m, n, col=True, verbose=False):
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

def round_1_column_routing(dst_column, verbose=False):
	# determine the intermediate_mapping: routing desitation in the first round
	if verbose:
		print("Round 1: column routing")
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32) - 1
	# initiatilize how many destinations are available
	# available_dst[i, j]: available dst in column i for dst j
	available_dst = np.zeros([n, n], dtype=np.int32)
	for i in range(m):
		for j in range(n):
			available_dst[j, dst_column[i, j]] += 1 
	# print(available_dst)
	# print(intermediate_mapping)
	for i in range(m):
		# build flow network
		# source 0
		# s: 1 ~ n
		# t: n+1 ~ 2n
		# sink: 2n+1
		G = nx.DiGraph()
		# add source and sink edges
		for j in range(n):
			G.add_edge(0, j+1, capacity=1, weight=1)
			G.add_edge(n+1+j, 2*n+1, capacity=1, weight=1)
		# add bipartitie connection
		for j in range(m):
			for k in range(n):
				if available_dst[k, dst_column[j, k]] > 0:
					G.add_edge(1+k, n+1+dst_column[j, k], capacity=1, weight=1)
		# nx.draw(G, with_labels = True)
		# plt.show()
		mincostFlow = nx.max_flow_min_cost(G, 0, 2*n+1)
		if verbose:
			print("~~~~~~~~~~~~~~~~~~~~~~~")
			print(mincostFlow)
		# find i-th matching: map the selected elements from current row to the i-th row
		for u in mincostFlow:
			if u == 0:
				continue
			vertices = mincostFlow[u]
			for v in vertices:
				if v == 2*n+1 or vertices[v] == 0:
					continue
				# edge (u, v) is in the matching, i.e., qbit will move from column u to v
				# find required source column
				required_src = u - 1 
				# find required destination column
				required_dst = v - (n + 1)
				# find row index in the required source column
				required_row_ind = np.where(dst_column[:, required_src] == required_dst)[0][0]
				# print("required_src = {}, required_dst = {}, required_row_ind = {}".format(required_src, required_dst, required_row_ind))
				# decrement available destination from src to dst
				available_dst[required_src, required_dst] -= 1
				# mark the corresponding data as mapped
				dst_column[required_row_ind, required_src] = -1
				# record the mapping
				intermediate_mapping[i, required_src] = required_row_ind
	# perform column routing
	swap_edges = []
	for i in range(n):
		swap_edges.append(line_routing(np.arange(m), intermediate_mapping[:, i]))
	if verbose:
		print("swap_edges = ")
		print(swap_edges)
		# adjust order
		print("parallelized swap_edges = ")
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	if verbose:
		print(p_swap_edges)
		print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping, p_swap_edges

def find_perfect_matching(dst_row, dst_column, current_row, matchings):
	# print("current_row = ", current_row)
	# print("dst_row = ", dst_row)
	# print("dst_column = ", dst_column)
	m, n = np.shape(dst_column)
	# print(m, n)
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
			G.add_edge(0, j+1, capacity=1, weight=1)
			G.add_edge(n+1+j, 2*n+1, capacity=1, weight=1)
		# add bipartitie connection
		for j in range(m):
			for k in range(n):
				if dst_column[j, k] != -1 and available_dst[k, dst_column[j, k]] > 0:
					G.add_edge(1+k, n+1+dst_column[j, k], capacity=1, weight=1)
		# nx.draw(G, with_labels = True)
		# plt.show()
		flowValue, maxFlow = nx.algorithms.flow.maximum_flow(G, 0, 2*n+1)
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
				if v == 2*n+1 or vertices[v] == 0:
					continue
				# edge (u, v) is in the matching, i.e., qbit will move from column u to v
				# find required source column
				required_src = u - 1 
				# find required destination column
				required_dst = v - (n + 1)
				# find row index in the required source column
				required_row_ind = np.where(dst_column[:, required_src] == required_dst)[0][0]
				# print("required_src = {}, required_dst = {}, required_row_ind = {}".format(required_src, required_dst, required_row_ind))
				# decrement available destination from src to dst
				available_dst[required_src, required_dst] -= 1
				# mark the corresponding data as mapped
				dst_column[required_row_ind, required_src] = -1
				# record matching [j, j', i, i']
				matching.append([required_src, required_dst, current_row + required_row_ind, dst_row[required_row_ind, required_src]])
		matchings.append(matching)

def round_1_column_routing_with_localism(dst_row, dst_column, verbose=False):
	# determine the intermediate_mapping: routing desitation in the first round
	if verbose:
		print("Round 1: column routing with localism")
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32) - 1
	matchings = []
	window_size = 1
	while len(matchings) < m and window_size <= m*2:
		# iterate over each slice
		start = 0
		for i in range(m // window_size + 1):
			end = np.min([start + window_size, m])
			if start < end:
				find_perfect_matching(dst_row[start:end], dst_column[start:end], start, matchings)
				start = end
		if verbose:
			print("window_size = {}, #matchings = {}".format(window_size, len(matchings)))
		window_size = 2*window_size
	assert(len(matchings) == m)
	if verbose:
		print("matchings: ")
		for match in matchings:
			print(match)
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
	if verbose:
		print("distance = \n", distance)
	bottleneck_matching = []
	sorted_distance = np.sort(distance.reshape([-1]))
	while sorted_distance.size != 0:
		mid = sorted_distance.size // 2
		delta = sorted_distance[mid]
		# print("distance computation:")
		# print(sorted_distance)
		# print(mid, delta)
		# tmp = distance.copy()
		# print(tmp)
		# tmp[tmp > delta] = -1
		# print(tmp)
		# create complete bipartite graph H
		H = nx.DiGraph()
		# build flow network
		# source 0
		# s: 1 ~ m
		# t: m+1 ~ 2m
		# sink: 2m+1
		# add source and sink edges
		for j in range(m):
			H.add_edge(0, j+1, capacity=1, weight=1)
			H.add_edge(m+1+j, 2*m+1, capacity=1, weight=1)
		# add bipartitie connection
		for j in range(m):
			for k in range(m):
				if distance[j, k] <= delta:
					H.add_edge(1+j, m+1+k, capacity=1, weight=distance[j, k])
		# find matching
		flowValue, maxFlow = nx.algorithms.flow.maximum_flow(H, 0, 2*m+1)
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
					if v == 2*m+1 or vertices[v] == 0:
						continue
					# find index of matching
					matching_ind = u - 1 
					# find index of row
					row_ind = v - (m + 1)
					bottleneck_matching.append([matching_ind, row_ind])
			sorted_distance = sorted_distance[:mid]
			# print("matching found")
			# print(bottleneck_matching)
	# test correct result
	if verbose:
		print(bottleneck_matching)
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
		swap_edges.append(line_routing(np.arange(m), intermediate_mapping[:, i]))
	if verbose:
		print("swap_edges = ")
		print(swap_edges)
		# adjust order
		print("parallelized swap_edges = ")
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	if verbose:
		print(p_swap_edges)
		print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping, p_swap_edges

# route dst_column to the correct place
def round_2_row_routing(dst_column, verbose=False):
	if verbose:
		print("Round 2: row routing")
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	swap_edges = []
	for i in range(m):
		swap_edges.append(line_routing(dst_column[i, :], np.arange(n)))
		intermediate_mapping[i, dst_column[i, :]] = np.arange(n)
	if verbose:
		print("swap_edges = ")
		print(swap_edges)
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n, False)
	if verbose:
		print("parallelized swap_edges = ")
		print(p_swap_edges)
		print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping, p_swap_edges

# rout dst_row to the correct place
def round_3_column_routing(dst_row, verbose=False):
	if verbose:
		print("Round 3: column routing")
	m, n = np.shape(dst_row)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	swap_edges = []
	for i in range(n):
		swap_edges.append(line_routing(dst_row[:, i], np.arange(m)))
		intermediate_mapping[dst_row[:, i], i] = np.arange(m)
	if verbose:
		print("swap_edges = ")
		print(swap_edges)
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	if verbose:
		print("parallelized swap_edges = ")
		print(p_swap_edges)
		print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping, p_swap_edges

def grid_route(src, dst, local=True, verbose=False):
	assert(len(src.shape) == 2)
	assert(len(src) == len(dst))
	# create mapping
	m, n = np.shape(src)
	arg_src = np.argsort(src.reshape([-1]))
	arg_dst = np.argsort(dst.reshape([-1]))
	if verbose:
		print(arg_src)
		print(arg_dst)
	mapping = np.zeros([m * n], dtype=np.int32)
	mapping[arg_src] = arg_dst
	print(mapping.reshape([m, n]))
	dst_column = mapping.reshape([m, n]) % n
	dst_row = mapping.reshape([m, n]) // n
	if verbose:
		print(dst_column)
		print(dst_row)
	# print(dst_column)
	# print(dst_row)
	if local:
		intermediate_mapping, p_swap_edge_1 = round_1_column_routing_with_localism(dst_row.copy(), dst_column.copy(), verbose)
	else:
		intermediate_mapping, p_swap_edge_1 = round_1_column_routing(dst_column.copy(), verbose)
	if verbose:
		print(intermediate_mapping)
	# swap dst_column and dst_row based on the intermediate_mapping
	for i in range(n):
		tmp = dst_column[:, i]
		dst_column[:, i] = tmp[intermediate_mapping[:, i]]
		tmp = dst_row[:, i]
		dst_row[:, i] = tmp[intermediate_mapping[:, i]]
	# print(dst_column)
	# print(dst_row)
	intermediate_mapping, p_swap_edge_2 = round_2_row_routing(dst_column.copy(), verbose)
	if verbose:
		print(intermediate_mapping)
	# swap dst_column and dst_row
	for i in range(m):
		tmp = dst_column[i, :]
		dst_column[i, :] = tmp[intermediate_mapping[i, :]]
		tmp = dst_row[i, :]
		dst_row[i, :] = tmp[intermediate_mapping[i, :]]
	# print(dst_column)
	# print(dst_row)
	intermediate_mapping, p_swap_edge_3 = round_3_column_routing(dst_row.copy(), verbose)
	if verbose:
		print(intermediate_mapping)
	# swap dst_column and dst_row based on the intermediate_mapping
	for i in range(n):
		tmp = dst_column[:, i]
		dst_column[:, i] = tmp[intermediate_mapping[:, i]]
		tmp = dst_row[:, i]
		dst_row[:, i] = tmp[intermediate_mapping[:, i]]
	if verbose:
		print(dst_column)
		print(dst_row)
	p_swap_edges = p_swap_edge_1 + p_swap_edge_2 + p_swap_edge_3
	if verbose:
		print("depth = {}".format(len(p_swap_edges)))
	return p_swap_edges

def grid_route_two_directions(src, dst, local=True, verbose=False):
	m, n = np.shape(src)
	swap_edges_1 = grid_route(src, dst, local, verbose)
	swap_edges_2 = grid_route(np.transpose(src), np.transpose(dst), local, verbose)
	if len(swap_edges_1) <= len(swap_edges_2):
		return swap_edges_1
	else:
		# transpose swap gates
		for lis in swap_edges_2:
			for pair in lis:
				pair[0] = (pair[0] % m) * n + (pair[0] // m)
				pair[1] = (pair[1] % m) * n + (pair[1] // m)
		return swap_edges_2

# test routing
# n1 = 6
# n2 = 6
# a = np.random.permutation(n1*n2).reshape([n1, n2])
# b = np.random.permutation(n1*n2).reshape([n1, n2])
# a = np.array([(0, 1), (2, 3)])
# b = np.array([(2, 0), (3, 1)])

# using Avah's example
# a = np.arange(15).reshape([5, 3])
# b = np.array([1, 5, 4, 0, 2, 3, 6, 10, 12, 13, 9, 7, 11, 14, 8]).reshape([5, 3])

# a = np.asarray([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(4, 4)
# b = np.asarray([ 1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 13, 12, 14, 15]).reshape(4, 4)
# a = np.arange(16).reshape(4, 4)
# b = np.asarray([14, 5, 7, 10, 1, 3, 6, 9, 2, 13, 11, 0, 4, 15, 8,
#                 12]).reshape(4, 4)
# a = np.arange(64).reshape(8, 8)
# b = np.asarray([
#     56, 58, 19, 39, 8, 27, 23, 48, 42, 11, 33, 30, 29, 21, 52, 46, 47, 22, 41,
#     63, 61, 26, 32, 13, 24, 43, 3, 2, 36, 35, 14, 6, 57, 4, 16, 25, 40, 51, 59,
#     18, 45, 44, 37, 50, 54, 60, 55, 15, 28, 34, 49, 10, 38, 20, 62, 5, 1, 53,
#     17, 0, 7, 9, 12, 31
# ]).reshape(8, 8)
a = np.asarray([0, 1, 3, 2, 4, 5,
				6, 14, 7, 9, 10, 11,
				12, 13, 8, 15, 16, 17,
				24, 19, 20, 21, 22, 23,
				18, 25, 26, 27, 28, 29,
				30, 31, 32, 33, 34, 35]).reshape(6, 6)
b = np.arange(36).reshape(6, 6)

print(a)
print(b)
import sys
# local = 0
# if len(sys.argv) > 1:
# 	local = int(sys.argv[1])
# swap_edges_nonlocal = grid_route(a, b, 0)
# swap_edges_nonlocal = grid_route_two_directions(a, b, 0)
# print("depth_nonlocal = ", len(swap_edges_nonlocal))
swap_edges_local = grid_route(a, b, 1, True)
# swap_edges_local = grid_route_two_directions(a, b, 1)
print("depth_local = ", len(swap_edges_local))

