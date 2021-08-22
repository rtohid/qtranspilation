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
		# nx.draw(G)
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
	print("parallelized swap_edges = ")
	print(parallelize_swap_gates(swap_edges, m, n))
	# adjust order
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
	print("parallelized swap_edges = ")
	print(parallelize_swap_gates(swap_edges, m, n, False))
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
	print("parallelized swap_edges = ")
	print(parallelize_swap_gates(swap_edges, m, n))
	return intermediate_mapping

def grid_route(src, dst):
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
a = np.random.permutation(12).reshape([3, 4])
print(a)
b = np.random.permutation(12).reshape([3, 4])
print(b)
grid_route(a, b)
