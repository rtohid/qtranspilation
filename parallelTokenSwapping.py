import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

def check_inverted(dst_x, dst_y):
	return (dst_x > dst_y)

# perform line routing
# @param src: source indices
# @param dst: destination indices
def line_routing(src, dst):
	assert(len(src.shape) == 1)
	assert(len(src) == len(dst))
	n = len(src)
	mapping = np.argsort(dst)
	print("start position  = {}".format(src))
	print("target position = {}".format(dst))
	start = 0
	current = src.copy()
	iteration = 0
	swap_edges = []
	while np.count_nonzero(current == dst) != n:
		# check edges
		swap_edge = []
		for i in range(start, n - 1, 2):
			if check_inverted(mapping[current[i]], mapping[current[i+1]]):
				# swap
				tmp = current[i]
				current[i] = current[i+1]
				current[i+1] = tmp
				swap_edge.append([i, i+1])
		swap_edges.append(swap_edge)
		# print("swap_edges in iteration {}: {}".format(iteration, swap_edge))
		# print("current position = {}".format(current))
		iteration += 1
		start = 1 - start
	print(swap_edges)		
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
	print("swap_edges = ")
	print(swap_edges)
	# adjust order
	print("parallelized swap_edges = ")
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	print(p_swap_edges)
	print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping

def find_perfect_matching(dst_row, dst_column, current_row, matchings):
	m, n = np.shape(dst_column)
	print(m, n)
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
				matching.append([required_src, required_dst, current_row + required_row_ind, current_row + dst_row[required_row_ind, required_src]])
		matchings.append(matching)

def round_1_column_routing_with_localism(dst_row, dst_column):
	# determine the intermediate_mapping: routing desitation in the first round
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
		print("window_size = {}, #matchings = {}".format(window_size, len(matchings)))
		window_size = 2*window_size
	assert(len(matchings) == m)
	print(matchings)
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
	print(distance)
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
			H.add_edge(0, j+1, capacity=1, weight=1)
			H.add_edge(m+1+j, 2*m+1, capacity=1, weight=1)
		# add bipartitie connection
		for j in range(m):
			for k in range(m):
				if distance[j, k] < delta:
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
	# test correct result
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
	print("swap_edges = ")
	print(swap_edges)
	# adjust order
	print("parallelized swap_edges = ")
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	print(p_swap_edges)
	print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping

# route dst_column to the correct place
def round_2_row_routing(dst_column):
	print("Round 2: row routing")
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	swap_edges = []
	for i in range(m):
		swap_edges.append(line_routing(dst_column[i, :], np.arange(n)))
		intermediate_mapping[i, dst_column[i, :]] = np.arange(n)
	print("swap_edges = ")
	print(swap_edges)
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n, False)
	print("parallelized swap_edges = ")
	print(p_swap_edges)
	print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping

# rout dst_row to the correct place
def round_3_column_routing(dst_row):
	print("Round 3: column routing")
	m, n = np.shape(dst_row)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	swap_edges = []
	for i in range(n):
		swap_edges.append(line_routing(dst_row[:, i], np.arange(m)))
		intermediate_mapping[dst_row[:, i], i] = np.arange(m)
	print("swap_edges = ")
	print(swap_edges)
	p_swap_edges = parallelize_swap_gates(swap_edges, m, n)
	print("parallelized swap_edges = ")
	print(p_swap_edges)
	print("depth = {}".format(len(p_swap_edges)))
	return intermediate_mapping

def grid_route(src, dst, local=False):
	assert(len(src.shape) == 2)
	assert(len(src) == len(dst))
	# create mapping
	m, n = np.shape(src)
	arg_src = np.argsort(src.reshape([-1]))
	arg_dst = np.argsort(dst.reshape([-1]))
	print(arg_src)
	print(arg_dst)
	mapping = np.zeros([m * n], dtype=np.int32)
	mapping[arg_src] = arg_dst
	dst_column = mapping.reshape([m, n]) % n
	dst_row = mapping.reshape([m, n]) // n
	print(dst_column)
	print(dst_row)
	# print(dst_column)
	# print(dst_row)
	if local:
		intermediate_mapping = round_1_column_routing_with_localism(dst_row.copy(), dst_column.copy())
	else:
		intermediate_mapping = round_1_column_routing(dst_column.copy())
	print(intermediate_mapping)
	# swap dst_column and dst_row based on the intermediate_mapping
	for i in range(n):
		tmp = dst_column[:, i]
		dst_column[:, i] = tmp[intermediate_mapping[:, i]]
		tmp = dst_row[:, i]
		dst_row[:, i] = tmp[intermediate_mapping[:, i]]
	# print(dst_column)
	# print(dst_row)
	intermediate_mapping = round_2_row_routing(dst_column.copy())
	print(intermediate_mapping)
	# swap dst_column and dst_row
	for i in range(m):
		tmp = dst_column[i, :]
		dst_column[i, :] = tmp[intermediate_mapping[i, :]]
		tmp = dst_row[i, :]
		dst_row[i, :] = tmp[intermediate_mapping[i, :]]
	# print(dst_column)
	# print(dst_row)
	intermediate_mapping = round_3_column_routing(dst_row.copy())
	print(intermediate_mapping)
	# swap dst_column and dst_row based on the intermediate_mapping
	for i in range(n):
		tmp = dst_column[:, i]
		dst_column[:, i] = tmp[intermediate_mapping[:, i]]
		tmp = dst_row[:, i]
		dst_row[:, i] = tmp[intermediate_mapping[:, i]]
	print(dst_column)
	print(dst_row)

# test routing
n1 = 6
n2 = 6
a = np.random.permutation(n1*n2).reshape([n1, n2])
b = np.random.permutation(n1*n2).reshape([n1, n2])
# using Avah's example
# a = np.arange(15).reshape([5, 3])
# b = np.array([1, 5, 4, 0, 2, 3, 6, 10, 12, 13, 9, 7, 11, 14, 8]).reshape([5, 3])
print(a)
print(b)
import sys
local = 0
if len(sys.argv) > 1:
	local = int(sys.argv[1])
grid_route(a, b, local)
