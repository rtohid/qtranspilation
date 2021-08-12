import numpy as np

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
	while np.count_nonzero(current == dst) != n:
		# check edges
		swap_edge = []
		for i in range(start, n - 1, 2):
			if check_inverted(mapping[current[i]], mapping[current[i+1]]):
				# swap
				tmp = current[i]
				current[i] = current[i+1]
				current[i+1] = tmp
				swap_edge.append([current[i], current[i+1]])
		print("swap_edges in iteration {}: {}".format(iteration, swap_edge))
		print("current position = {}".format(current))
		iteration += 1
		start = 1 - start		

def round_1_column_routing(dst_column):
	# determine the intermediate_mapping: routing desitation in the first round
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	# initiatilize how many destinations are available
	# available_dst[i, j]: available dst in column i for dst j
	available_dst = np.zeros([n, n], dtype=np.int32)
	for i in range(m):
		for j in range(n):
			available_dst[j, dst_column[i, j]] += 1 
	# print(available_dst)
	# compute intermediate_mapping destination
	for i in range(m):
		avail = np.zeros([n], dtype=np.int32)
		for j in range(n):
			avail += available_dst[:, j]
			# map the maximal available dst to the i-th row
			required_dst = np.argmax(avail)
			required_ind = np.where(dst_column[:, j] == required_dst)[0][0]
			dst_column[i, j] = -1
			available_dst[required_dst, j] -= 1
			avail[required_dst] = -(m+1)
			intermediate_mapping[required_ind, j] = i
			# print("required_dst = {}, required_ind = {}".format(required_dst, required_ind))
	# print(intermediate_mapping)
	# perform column routing
	for i in range(n):
		line_routing(np.arange(m), intermediate_mapping[:, i])
	return intermediate_mapping

# route dst_column to the correct place
def round_2_row_routing(dst_column):
	m, n = np.shape(dst_column)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	for i in range(m):
		line_routing(dst_column[i, :], np.arange(n))
		intermediate_mapping[i, dst_column[i, :]] = np.arange(n)
	return intermediate_mapping

# rout dst_row to the correct place
def round_3_column_routing(dst_row):
	m, n = np.shape(dst_row)
	intermediate_mapping = np.zeros([m, n], dtype=np.int32)
	for i in range(n):
		line_routing(dst_row[:, i], np.arange(m))
		intermediate_mapping[dst_row[:, i], i] = np.arange(n)
	return intermediate_mapping


def grid_route(src, dst):
	assert(len(src.shape) == 2)
	assert(len(src) == len(dst))
	# create mapping
	m, n = np.shape(src)
	mapping = np.argsort(dst.reshape([-1]))
	mapping_column = mapping % n
	mapping_row = mapping // m
	dst_column = mapping_column[src]
	dst_row = mapping_row[src]
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
a = np.arange(16).reshape([4, 4])
b = 15 - a 
grid_route(a, b)
